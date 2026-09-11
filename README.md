# EchoJEPA UI

Streamlit app for LV segmentation and LVEF prediction from an echocardiography video.

## Structure

```text
ECHO_UI/
├── app.py                  # Streamlit application
├── run_seg.py              # EchoJEPA segmentation model
├── run_reg.py              # LVEF regression head
├── demo_samples.csv        # Optional ground-truth manifest
├── models/                 # Backbone, decoder, and regression checkpoints
├── uploads/                # Uploaded videos
└── outputs/                # Generated videos and metrics
```

## Install

```bash
git clone git@github.com:harshit-8118/ECHO_UI.git
cd ECHO_UI
git lfs pull
pip install -r requirements.txt
```

Install FFmpeg separately for browser-compatible output video encoding.

## Run

```bash
streamlit run app.py
```

Open the displayed URL, upload a video, and select **Run analysis**.
