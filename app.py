"""Streamlit UI for the standalone EchoJEPA segmentation runner.

Start from this folder with:
    streamlit run app.py
"""

from __future__ import annotations

import csv
import base64
import hashlib
import html
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch

from run_seg import (
    DEFAULT_BACKBONE,
    DEFAULT_SEGMENTATION,
    RESOLUTION,
    load_model,
    make_clip,
    overlay,
)
from run_reg import load_regression_probe, predict_ef


HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

UPLOAD_DIR = HERE / "uploads"
OUTPUT_ROOT = HERE / "outputs"
REGRESSION_FP16 = MODELS_DIR / "reg_head_best.pt"
DEMO_MANIFEST = HERE / "demo_samples.csv"

@st.cache_resource(show_spinner="Loading EchoJEPA backbone and segmentation decoder...")
def get_model(backbone_path: str, segmentation_path: str, device_name: str):
    """Load the large model once and reuse it across Streamlit reruns."""
    return load_model(Path(backbone_path), Path(segmentation_path), torch.device(device_name))


@st.cache_resource(show_spinner="Loading EF regression probe...")
def get_regression_probe(regression_path: str, device_name: str):
    return load_regression_probe(Path(regression_path), torch.device(device_name))


def save_uploaded_video(uploaded_file) -> Path:
    """Save an uploaded video under a stable content-derived filename."""
    phase_start = time.perf_counter()
    content = uploaded_file.getvalue()
    digest = hashlib.sha256(content).hexdigest()[:12]
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{Path(uploaded_file.name).stem}_{digest}{suffix}"
    if not path.exists():
        path.write_bytes(content)
    print(f"[TIMING] Save/read uploaded file: {time.perf_counter() - phase_start:.3f}s ({len(content) / 1e6:.1f} MB)")
    return path


def open_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Write an intermediate MP4 without probing unavailable H.264 hardware."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the intermediate mp4v video.")
    print(f"[VIDEO] Writing intermediate {path.name} with mp4v at {width}x{height}")
    return writer


def make_browser_compatible(intermediate: Path, output: Path) -> None:
    """Use CPU libx264 when ffmpeg exists; otherwise retain the mp4v file."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(intermediate),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and output.is_file():
            intermediate.unlink(missing_ok=True)
            print(f"[VIDEO] Encoded browser-compatible H.264 with CPU libx264: {output.name}")
            return
        print(f"[VIDEO] ffmpeg/libx264 failed; retaining mp4v. Details: {result.stderr.strip()}")
    else:
        print("[VIDEO] ffmpeg executable not found; retaining mp4v output.")

    if output.exists():
        output.unlink()
    intermediate.replace(output)


def read_video_for_display_and_model(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Decode original RGB frames and make separate square model inputs."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    original_frames: list[np.ndarray] = []
    model_frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_frames.append(rgb)
        model_frames.append(cv2.resize(rgb, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR))
    capture.release()
    if not original_frames:
        raise RuntimeError(f"No frames were decoded from {path}")
    return original_frames, model_frames, fps


def show_looping_video(path: Path, label: str, aspect_ratio: float) -> None:
    """Render an autoplaying, muted, cyclic MP4 while preserving its ratio."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    safe_label = html.escape(label)
    height = max(240, min(620, int(650 / max(aspect_ratio, 0.1))))
    components.html(
        f"""
        <video aria-label="{safe_label}" autoplay loop muted controls playsinline
               style="display:block;width:100%;height:auto;max-height:{height}px;background:#000;object-fit:contain;">
          <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
          Your browser does not support MP4 video.
        </video>
        """,
        height=height,
        scrolling=False,
    )


@torch.inference_mode()
def process_video(
    video_path: Path,
    model,
    device: torch.device,
    threshold: float,
    frame_step: int,
    progress_bar,
) -> tuple[tuple[Path, Path, Path], list[np.ndarray]]:
    """Create equally sized original and segmentation-overlay videos."""
    total_start = time.perf_counter()
    phase_start = time.perf_counter()
    original_frames, frames, fps = read_video_for_display_and_model(video_path)
    decode_seconds = time.perf_counter() - phase_start
    print(f"[TIMING] Decode and resize video: {decode_seconds:.2f}s ({len(frames)} frames)")
    output_dir = OUTPUT_ROOT / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "original.mp4"
    segmentation_path = output_dir / "segmentation_overlay.mp4"
    original_intermediate = output_dir / "original_intermediate.mp4"
    segmentation_intermediate = output_dir / "segmentation_intermediate.mp4"
    metrics_path = output_dir / "frame_metrics.csv"

    phase_start = time.perf_counter()
    output_fps = fps / frame_step
    original_height, original_width = original_frames[0].shape[:2]
    original_writer = open_video_writer(original_intermediate, output_fps, original_width, original_height)
    segmentation_writer = open_video_writer(
        segmentation_intermediate, output_fps, original_width, original_height
    )
    print(f"[TIMING] Initialize video writers: {time.perf_counter() - phase_start:.3f}s")
    metrics: list[dict[str, int | float]] = []
    clip_seconds = 0.0
    prediction_seconds = 0.0
    render_seconds = 0.0
    write_seconds = 0.0
    frame_indices = range(0, len(frames), frame_step)
    prediction_count = len(frame_indices)

    try:
        for output_index, frame_index in enumerate(frame_indices):
            frame = frames[frame_index]
            original_frame = original_frames[frame_index]
            phase_start = time.perf_counter()
            model_dtype = next(model.parameters()).dtype
            clip = make_clip(frames, frame_index).unsqueeze(0).to(
                device=device, dtype=model_dtype, non_blocking=True
            )
            clip_seconds += time.perf_counter() - phase_start

            phase_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                probability = torch.sigmoid(model(clip))[0, 0].float().cpu().numpy()
            # The CPU transfer above synchronizes CUDA, so this includes actual GPU inference time.
            prediction_seconds += time.perf_counter() - phase_start

            phase_start = time.perf_counter()
            mask = probability >= threshold
            # Prediction stays at model resolution; resize only the binary mask
            # back to the source frame without distorting the displayed video.
            display_mask = cv2.resize(
                mask.astype(np.uint8),
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            original_bgr = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.cvtColor(overlay(original_frame, display_mask), cv2.COLOR_RGB2BGR)
            render_seconds += time.perf_counter() - phase_start

            phase_start = time.perf_counter()
            original_writer.write(original_bgr)
            segmentation_writer.write(overlay_bgr)
            write_seconds += time.perf_counter() - phase_start
            metrics.append(
                {
                    "frame": frame_index,
                    "mask_area_pixels": int(mask.sum()),
                    "mask_area_fraction": float(mask.mean()),
                    "mean_probability": float(probability.mean()),
                }
            )
            processed = output_index + 1
            progress_bar.progress(processed / prediction_count, text=f"Segmenting {processed}/{prediction_count}")
            if processed % 25 == 0 or processed == prediction_count:
                print(
                    f"[PROGRESS] {processed}/{prediction_count} | "
                    f"clip={clip_seconds:.2f}s prediction={prediction_seconds:.2f}s "
                    f"render={render_seconds:.2f}s write={write_seconds:.2f}s"
                )
    finally:
        original_writer.release()
        segmentation_writer.release()

    phase_start = time.perf_counter()
    make_browser_compatible(original_intermediate, original_path)
    make_browser_compatible(segmentation_intermediate, segmentation_path)
    transcode_seconds = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    csv_seconds = time.perf_counter() - phase_start
    total_seconds = time.perf_counter() - total_start
    print("\n[TIMING SUMMARY]")
    print(f"  Video decode/resize : {decode_seconds:8.2f}s")
    print(f"  Clip preparation    : {clip_seconds:8.2f}s")
    print(f"  Model prediction    : {prediction_seconds:8.2f}s")
    print(f"  Mask/overlay render : {render_seconds:8.2f}s")
    print(f"  VideoWriter encoding: {write_seconds:8.2f}s")
    print(f"  Browser transcoding : {transcode_seconds:8.2f}s")
    print(f"  CSV writing         : {csv_seconds:8.3f}s")
    print(f"  Total processing    : {total_seconds:8.2f}s")
    print(f"  Average/prediction  : {prediction_seconds / prediction_count:8.3f}s")
    print(f"  Throughput          : {prediction_count / total_seconds:8.2f} predictions/s\n")
    return (original_path, segmentation_path, metrics_path), frames


def main() -> None:
    st.set_page_config(page_title="EchoJEPA Analysis", page_icon="🫀", layout="wide")
    st.title("EchoJEPA LV Analysis")
    st.caption("Upload one video to predict LVEF and visualize the LV segmentation.")

    with st.sidebar:
        st.header("Model")
        st.caption(f"Backbone: `{DEFAULT_BACKBONE.name}`")
        st.caption(f"Segmentation: `{DEFAULT_SEGMENTATION.name}`")
        st.caption(f"EF regression: `{REGRESSION_FP16.name}`")
        threshold = st.slider("Mask threshold", 0.0, 1.0, 0.5, 0.01)
        frame_step = st.select_slider(
            "Process every Nth frame",
            options=[1, 2, 4, 8],
            value=1,
            help="Higher values are faster but skip intermediate target frames.",
        )
        ef_num_clips = st.select_slider(
            "EF inference clips",
            options=[1, 3, 5],
            value=1,
            help="One deterministic uniform clip, plus randomized stratified clips when using 3 or 5.",
        )
        requested_device = st.selectbox(
            "Device",
            ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
        )
        show_ground_truth = st.radio(
            "Show ground-truth LVEF",
            ["Off", "On"],
            horizontal=True,
        ) == "On"
        st.caption("The model always uses centered 16-frame clips around each selected target frame.")

    uploaded_file = st.file_uploader("Choose an AVI or MP4 video", type=["avi", "mp4", "mov", "mkv"])
    if uploaded_file is None:
        st.info("Upload a video to begin.")
        return

    missing = [path for path in (DEFAULT_BACKBONE, DEFAULT_SEGMENTATION, REGRESSION_FP16) if not path.is_file()]
    if missing:
        st.error("Missing checkpoint:\n\n" + "\n\n".join(map(str, missing)))
        return

    video_path = save_uploaded_video(uploaded_file)
    true_ef = None
    if show_ground_truth:
        if DEMO_MANIFEST.is_file():
            with DEMO_MANIFEST.open(newline="") as file:
                rows = list(csv.DictReader(file))
            uploaded_name = Path(uploaded_file.name).stem
            match = next((row for row in rows if Path(row["file_name"]).stem == uploaded_name), None)
            if match:
                true_ef = float(match["true_ef"])
            else:
                st.warning(f"Ground-truth LVEF is unavailable for `{uploaded_name}`.")
        else:
            st.warning(f"Ground-truth manifest not found: {DEMO_MANIFEST}")
    result_key = (
        f"{video_path}:{threshold}:{frame_step}:{ef_num_clips}:"
        f"{DEFAULT_BACKBONE}:{DEFAULT_SEGMENTATION}:{REGRESSION_FP16}"
    )
    if st.button("Run analysis", type="primary", use_container_width=True):
        try:
            device = torch.device(requested_device)
            phase_start = time.perf_counter()
            model = get_model(str(DEFAULT_BACKBONE), str(DEFAULT_SEGMENTATION), requested_device)
            print(f"[TIMING] Streamlit model fetch (load or cache): {time.perf_counter() - phase_start:.2f}s")
            progress = st.progress(0.0, text="Preparing video...")
            result_paths, regression_frames = process_video(
                video_path, model, device, threshold, frame_step, progress
            )
            ef_start = time.perf_counter()
            regression_probe = get_regression_probe(str(REGRESSION_FP16), requested_device)
            ef_result = predict_ef(
                model.encoder,
                regression_probe,
                regression_frames,
                device,
                num_clips=ef_num_clips,
            )
            print(f"[TIMING] EF regression total: {time.perf_counter() - ef_start:.2f}s")
            print(f"[EF] {ef_result}")
            progress.empty()
            st.session_state["segmentation_result"] = (
                result_key,
                *map(str, result_paths),
                ef_result,
            )
        except Exception as error:
            st.exception(error)
            return

    saved_result = st.session_state.get("segmentation_result")
    if not saved_result or saved_result[0] != result_key:
        st.info("Press **Run analysis** to process this video.")
        return

    _, original_path_text, segmentation_path_text, metrics_path_text, ef_result = saved_result
    original_path = Path(original_path_text)
    segmentation_video_path = Path(segmentation_path_text)
    metrics_path = Path(metrics_path_text)
    if not all(path.is_file() for path in (original_path, segmentation_video_path, metrics_path)):
        st.warning("Saved results are no longer available. Run segmentation again.")
        return

    st.success("Analysis complete")
    ef_value = float(ef_result["ef_percent"])
    ef_std = float(ef_result["ef_std"])
    if true_ef is None:
        st.metric(
            "Predicted LVEF",
            f"{ef_value:.2f}%",
            help=f"EF clip standard deviation: {ef_std:.3f} points.",
        )
    else:
        correct_col, predicted_col = st.columns(2)
        correct_col.metric("Ground-truth LVEF", f"{true_ef:.2f}%")
        predicted_col.metric(
            "Predicted LVEF",
            f"{ef_value:.2f}%",
            delta=f"{ef_value - true_ef:+.2f} points",
            delta_color="inverse",
            help=f"EF clip standard deviation: {ef_std:.3f} points.",
        )
    original_column, segmentation_column = st.columns(2)
    video_aspect_ratio = 1.0
    capture = cv2.VideoCapture(str(original_path))
    if capture.isOpened():
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if height > 0:
            video_aspect_ratio = width / height
    capture.release()
    with original_column:
        st.subheader("Original video")
        show_looping_video(original_path, "Original video", video_aspect_ratio)
    with segmentation_column:
        st.subheader("Segmentation mask overlay")
        show_looping_video(segmentation_video_path, "Segmentation mask overlay", video_aspect_ratio)

    with metrics_path.open("rb") as file:
        st.download_button(
            "Download frame metrics CSV",
            data=file.read(),
            file_name=f"{video_path.stem}_frame_metrics.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
