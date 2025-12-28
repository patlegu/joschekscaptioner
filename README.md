# Joschek's Captioner

Another AI training tool. This one is "vibe coded", which means I wrote it until it worked for me. If it works for you, that's a happy accident.

## Features
- **Model Selector**: Because we all know your file organization is a mess. Point it to wherever you hid your vision models this time.
- **Batch Captioning**: Queue up folders to caption while you go contemplate what you are doing.
- **Cropping**: Uses YOLOv8 to find people and crop them. It sometimes works.
- **Caption Editor**: A groundbreaking text box to fix the AI's hallucinations. With filter function.
- **Problem Bin**: One-click functionality to yeet problematic pairs into a separate folder so you can deal with them "later" (never).

## Installation

You need Python 3.10+ and an NVIDIA GPU. If you don't have a GPU, enjoy waiting.

### Arch / Manjaro (btw)
```bash
sudo pacman -Syu python-tk zenity nvidia-utils git
git clone https://github.com/realjoschek/joschekscaptioner.git
cd joschekscaptioner
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Debian / Ubuntu / Mint
```bash
sudo apt update && sudo apt install -y python3-tk zenity git
git clone https://github.com/realjoschek/joschekscaptioner.git
cd joschekscaptioner
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows (I'm sorry)
1. Install Python 3.10+ (Add to PATH, don't ask why).
2. Install Git.
3. Paste this into PowerShell and pray:
   ```powershell
   git clone https://github.com/realjoschek/joschekscaptioner.git
   cd joschekscaptioner
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Setup llama-server
You need `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp)
1. **Get the binary**: Put `llama-server` (or `.exe`) in root or `./build/bin/`.
2. **Get a model**:
   - Recommended: [Qwen3-VL-8B-Abliterated-Caption-it](https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it). If you are planning on captioning naughty stuff.
   - Alternative: Qwen VL 3 - standart. better sometimes.
   - Don't forget the mmproj file.

## Usage

**Linux:**
```bash
source venv/bin/activate
python joschekscaptions.py
```

**Windows:**
```powershell
.\venv\Scripts\activate
python joschekscaptions.py
```

1. **Server Tab**: Pick your binary and model. Hit start. There's a "Kill GPU processes" button for when things go south.
2. **Batch Tab**: Point it at images. Wait.
3. **Editor Tab**: Fix the captions.
4. **Crop Humans**: Automagical cropping.

## License
GPL
