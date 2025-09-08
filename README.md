---
title: Clothing Classifier API
emoji: 🐢
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
license: mit
short_description: 'Classify clothing images using a trained PyTorch model.  '
---

# Clothing Classifier API

Classify clothing images using a trained PyTorch model.  

## Features
- Pretrained ResNet18 (or simple CNN)
- FastAPI endpoint: `/predict` (POST image file)
- Gradio web interface for easy uploads

## Quick Start

### Local
```bash
git clone <repo>
cd clothing-classifier-api
pip install -r requirements.txt
python main.py