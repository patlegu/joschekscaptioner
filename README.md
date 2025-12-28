# Joschek's Captioner

A vibe coded image captioning tool for AI training datasets.

## Features
- **Batch Captioning**: Uses llama-server (llama.cpp) and OpenAI-compatible API to caption images.
- **Human Cropping**: Detects people using YOLOv8-seg and crops them at optimal resolutions (768, 1024, 1536, 2048) without resizing.
- **Native UI**: Built with Tkinter, featuring a native folder picker (Zenity fallback).
- **Caption Editor**: Side-by-side image and text editor for quick dataset refinement.

## Installation

### 1. Prerequisites
Ensure you have Python 3.10+ and the following system packages:
```bash
sudo pacman -Syu
sudo pacman -S python-tk zenity nvidia-utils
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python numpy openai pillow
```

### 3. Setup llama-server (for Captioning)
You need `llama-server` from the [llama.cpp](https://github.com/ggerganov/llama.cpp) project and a Vision-capable model (like LLaVA).
1. Download/Build `llama-server`.
2. Download a GGUF model and its mmproj projector file.

## Usage
Run the application:
```bash
python joschek_captioner_v11.py
```

1. **Server Tab**: Configure and start your local llama-server.
2. **Batch Tab**: Add folders to the queue and generate captions.
3. **Editor Tab**: Manually review and edit generated captions.
4. **Crop Humans Tab**: Select input/output folders to automatically detect and crop people from images at the best fitting resolution (768, 1024, 1536, or 2048).

### Recomended Models
qwen-vl-3-abliterated-caption-it

## License
gpl
