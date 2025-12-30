Joschek's Captioner (v27)
(Insert your screenshot here)

A vibe-coded image dataset caption tool that started as a "happy accident" and evolved into a robust multi-backend Swiss Army Knife. It does exactly what I need, and now it might actually work on your machine without crashing.

## What's new in v27?
We moved from "it works on my machine" to "it works on yours too".

- Multi-Backend Support: Use Ollama, LM Studio, or the native llama-server. Your choice.
- Config-First: A human-readable joschek_captioner.json file is generated so you can tweak settings without touching the code.
- Strict Architecture: The code was refactored from a spaghetti script to a typed, threaded application. It doesn't freeze anymore when you look at it wrong.

## Features
- Engine Selector: Because running raw binaries isn't for everyone. Toggle between Native, Ollama, and LM Studio instantly.
- Batch Captioning: Queue up folders. Go touch grass. Come back to finished text files.
- Smart Cropping: Uses YOLOv8 to detect humans and crop them. It works surprisingly well.
- Lazy Editor: A visual editor that handles folders with thousands of images without eating all your RAM.
- Filter & Move: One-click functionality to yeet images containing specific keywords into a separate folder.
- Config Profiles: Save your "Realistic" settings and your "Anime" settings separately.

## Installation
You need Python 3.10+ and preferably an NVIDIA GPU (unless you like waiting).

### Arch / Manjaro (btw)
```Bash
sudo pacman -Syu python-tk zenity nvidia-utils git
git clone https://github.com/realjoschek/joschekscaptioner.git
cd joschekscaptioner
python -m venv venv
source venv/bin/activate
pip install openai pillow opencv-python numpy ultralytics requests
```

### Debian / Ubuntu / Mint
```Bash
sudo apt update && sudo apt install -y python3-tk zenity git
git clone https://github.com/realjoschek/joschekscaptioner.git
cd joschekscaptioner
python3 -m venv venv
source venv/bin/activate
pip install openai pillow opencv-python numpy ultralytics requests
```

### Windows (I'm sorry)
1. Install Python 3.10+ (Add to PATH, don't ask why).
2. Install Git.
3. Paste this into PowerShell and pray:
``` PowerShell
git clone https://github.com/realjoschek/joschekscaptioner.git
cd joschekscaptioner
python -m venv venv
.\venv\Scripts\activate
pip install openai pillow opencv-python numpy ultralytics requests
```

## Usage Roadmap
1. **Choose your Engine** (Server Tab)
The app needs a brain.  
Pick one:
- **Option A**: Ollama (Easiest)  
Run ollama serve in a terminal.  
In the app, select Backend: Ollama.  
Click Refresh Models -> Select one -> Connect.
- **Option B**: LM Studio  
Start the Local Server in LM Studio (Port 1234).  
In the app, select Backend: LM Studio -> Connect.
- **Option C**: Native  
Point the app to your llama-server binary and a .gguf model.  
Click Start Server.

2. **Batch Captioning**
Go to Batch Captioning.  
Add folders.  
Write a system prompt (or use the default).  
Hit Start Processing.  

3. **Review & Edit**  
Go to Manual Edit.  
Load your folder.  
Scroll (lazy loading is active).  
Edit text (auto-saves).  
Double-click to Zoom.  

4. **Advanced**: Configuration File  
On first launch, the app creates joschek_captioner.json in your user config folder. You can open this file with a   text editor to enforce defaults (backend choice, ports, prompts) using comments.  

```JSON

{
    # Uncomment to force Ollama backend on startup
    # "backend": "Ollama",
    "port": "11434"
}
```
### Recommended Models
Ollama: [llava:v1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf), [moondream](https://moondream.ai/), or [llama3.2-vision](https://ollama.com/library/llama3.2-vision).  
Native: [Qwen3-VL-8B-Abliterated-Caption-it](https://huggingface.co/prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it) (still the goat for uncensored captioning).

License
GPL