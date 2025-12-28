#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joschek’s Captioner  v22-Linux-only
Light-weight pure-Tk folder picker – no crashes, no external deps
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import os
import shutil
import threading
import signal
import glob
import base64
import json
import numpy as np
import cv2
from pathlib import Path
from openai import OpenAI
from PIL import Image, ImageTk

# Optional YOLO import
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ---------------- CONFIG ----------------
CONFIG_FILE = Path.home() / ".config" / "joschek_captioner.json"
DEFAULT_PORT = "11434"
DEFAULT_CTX  = "8192"
DEFAULT_BATCH= "512"
DEFAULT_GPU  = "99"
API_URL      = f"http://localhost:{DEFAULT_PORT}/v1"
DEFAULT_PROMPT = "Describe this image in detail for an AI training dataset. Focus on clothing, background, textures, and lighting."
TARGETS = [768, 1024, 1536, 2048]

# ---------------- PALETTE ----------------
BG = "#2b2e37"
CARD = "#353945"
INPUT = "#3d424e"
TEXT = "#d3dae3"
DIM  = "#7c818c"
BORDER = BG
BLUE = "#5294e2"
GREEN= "#73d216"
RED  = "#cc0000"

# ---------------- UTILS ----------------
class Config:
    def __init__(self):
        self.config_dir = CONFIG_FILE.parent
        self.data = self.load()
    def load(self):
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            if CONFIG_FILE.exists():
                return json.loads(CONFIG_FILE.read_text())
        except Exception as e:
            print("Config load error:", e)
        return {
            "server_binary": "./build/bin/llama-server",
            "model_file": "",
            "projector_file": "",
            "port": DEFAULT_PORT,
            "context": DEFAULT_CTX,
            "gpu_layers": DEFAULT_GPU,
            "last_prompt": DEFAULT_PROMPT
        }
    def save(self):
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.write_text(json.dumps(self.data, indent=2))
        except Exception as e:
            print("Config save error:", e)
    def get(self, key, default=None):
        return self.data.get(key, default)
    def set(self, key, value):
        self.data[key] = value
        self.save()

# ---------------- WIDGETS ----------------
class QueueItem(tk.Frame):
    def __init__(self, parent, path: Path, remove_cb, config):
        super().__init__(parent, bg=CARD)
        self.folder_path = path
        self.status = "pending"
        self.remove_cb = remove_cb
        self.config = config
        main = tk.Frame(self, bg=CARD)
        main.pack(fill="both", expand=True, padx=14, pady=10)
        header = tk.Frame(main, bg=CARD)
        header.pack(fill="x", pady=(0, 6))
        tk.Label(header, text=self.folder_path.name, bg=CARD, fg=TEXT,
                 font=("Sans", 9), anchor="w").pack(side="left", fill="x", expand=True)
        self.status_lbl = tk.Label(header, text="Ready", bg=CARD, fg=DIM, font=("Sans", 8))
        self.status_lbl.pack(side="left", padx=8)
        close = tk.Label(header, text="×", bg=CARD, fg=DIM, font=("Sans", 14), cursor="hand2")
        close.pack(side="right")
        close.bind("<Button-1>", lambda e: remove_cb(self))
        close.bind("<Enter>", lambda e: close.config(fg=RED))
        close.bind("<Leave>", lambda e: close.config(fg=DIM))
        tk.Label(main, text=str(self.folder_path), bg=CARD, fg=DIM, font=("Sans", 7), anchor="w").pack(fill="x", pady=(0, 8))
        self.prompt = tk.Text(main, height=2, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                              font=("Sans", 8), insertbackground=BLUE, wrap="word")
        self.prompt.insert("1.0", config.get("last_prompt", DEFAULT_PROMPT))
        self.prompt.bind("<KeyRelease>", lambda e: config.set("last_prompt", self.get_prompt()))
        self.prompt.pack(fill="x")
    def set_status(self, state, msg=""):
        color = {"processing": BLUE, "done": GREEN, "error": RED}.get(state, DIM)
        self.status_lbl.config(text=msg, fg=color)
    def get_prompt(self):
        return self.prompt.get("1.0", "end-1c").strip()

class ScrollFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.content = tk.Frame(canvas, bg=BG)
        self.content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))

# ---------------- CROP WORKER ----------------
class CropWorker:
    def __init__(self, input_path, output_path, target_res, model, update_progress, update_log, finished_cb):
        self.input_path = input_path
        self.output_path = output_path
        self.target_res = target_res
        self.model = model
        self.update_progress = update_progress
        self.update_log = update_log
        self.finished_cb = finished_cb
        self.running = True

    def run(self):
        try:
            os.makedirs(self.output_path, exist_ok=True)
            
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.WEBP']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(self.input_path, ext)))
            
            total_files = len(files)
            if total_files == 0:
                self.update_log("No images found in folder.")
                self.finished_cb()
                return

            self.update_log(f"Found {total_files} images to process.")
            self.model.fuse()

            for i, f in enumerate(files):
                if not self.running: break

                filename = os.path.basename(f)
                name_no_ext = os.path.splitext(filename)[0]
                self.update_log(f"Processing {filename}...")

                image = cv2.imread(f)
                if image is None:
                    continue

                try:
                    results = self.model.predict(image, conf=0.5, classes=[0], verbose=False)
                except Exception as e:
                    self.update_log(f"Model error on {filename}: {e}")
                    continue

                if results[0].masks is None or len(results[0].masks) == 0:
                    continue

                img_h, img_w = image.shape[:2]

                for mask_idx, mask in enumerate(results[0].masks):
                    if not self.running: break

                    points = mask.xy[0].astype(np.int32)
                    x, y, w, h = cv2.boundingRect(points)
                    
                    # Add a small safe margin around the detected object
                    SAFE_MARGIN = 10
                    x = max(0, x - SAFE_MARGIN)
                    y = max(0, y - SAFE_MARGIN)
                    w = min(img_w - x, w + SAFE_MARGIN * 2)
                    h = min(img_h - y, h + SAFE_MARGIN * 2)

                    # Determine crop size based on TARGETS [768, 1024, 1536, 2048]
                    # it should not resize the images but try to crop to the best fitting sizes
                    longest_side = max(w, h)
                    
                    # Find the "best fitting" target from TARGETS
                    if self.target_res == "KEEP":
                        best_target = longest_side
                    elif isinstance(self.target_res, int):
                        best_target = self.target_res
                    else: # AUTO
                        best_target = TARGETS[0]
                        min_diff = abs(longest_side - TARGETS[0])
                        for t in TARGETS:
                            diff = abs(longest_side - t)
                            if diff < min_diff:
                                min_diff = diff
                                best_target = t
                    
                    if w >= h:
                        crop_w = best_target
                        crop_h = int(best_target * (h / w))
                    else:
                        crop_h = best_target
                        crop_w = int(best_target * (w / h))

                    # Center the crop on the detected object
                    cx = x + (w // 2)
                    cy = y + (h // 2)
                    
                    crop_x1 = cx - (crop_w // 2)
                    crop_y1 = cy - (crop_h // 2)
                    
                    # Adjust to stay within image bounds
                    if crop_x1 < 0: crop_x1 = 0
                    if crop_y1 < 0: crop_y1 = 0
                    
                    crop_x2 = crop_x1 + crop_w
                    crop_y2 = crop_y1 + crop_h
                    
                    if crop_x2 > img_w:
                        crop_x2 = img_w
                        crop_x1 = max(0, crop_x2 - crop_w)
                    if crop_y2 > img_h:
                        crop_y2 = img_h
                        crop_y1 = max(0, crop_y2 - crop_h)

                    final_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    actual_h, actual_w = final_crop.shape[:2]
                    res_tag = f"{max(actual_w, actual_h)}px"
                    save_path = os.path.join(self.output_path, f"{name_no_ext}_human_{mask_idx}_{res_tag}.jpg")
                    cv2.imwrite(save_path, final_crop)

                self.update_progress(i + 1, total_files)
            
            self.update_log("Done.")
        except Exception as e:
            self.update_log(f"Error: {e}")
        finally:
            self.finished_cb()

# ---------------- MAIN APP ----------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Joschek's Captioner v22-Linux")
        root.geometry("1100x720")
        root.configure(bg=BG)
        self.config = Config()
        self.setup_styles()
        self.server_proc = None
        self.batch_running = False
        self.queue = []
        self.client = None
        self.current_editor_folder = None
        self.editor_items = []
        self.thumb_size = 128
        # notebook
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)
        self.tab_srv   = tk.Frame(nb, bg=BG)
        self.tab_batch = tk.Frame(nb, bg=BG)
        self.tab_editor= tk.Frame(nb, bg=BG)
        self.tab_filter= tk.Frame(nb, bg=BG)
        self.tab_crop  = tk.Frame(nb, bg=BG)
        nb.add(self.tab_srv,   text="Server")
        nb.add(self.tab_batch, text="Batch")
        nb.add(self.tab_editor,text="Editor")
        nb.add(self.tab_filter,text="Filter & Move")
        nb.add(self.tab_crop,  text="Crop Humans")
        self.build_server()
        self.build_batch()
        self.build_editor()
        self.build_filter()
        self.build_crop()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
    # ---------------- STYLES (ELEGANT) ----------------
    def setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
        s.configure("TNotebook.Tab", background=CARD, foreground=TEXT,
                    padding=[26, 12], borderwidth=0, font=("Sans", 10))
        s.map("TNotebook.Tab", background=[("selected", INPUT)], foreground=[("selected", TEXT)])
        s.configure("TProgressbar", background=BLUE, troughcolor=BG, borderwidth=0, thickness=4)
        s.configure("Vertical.TScrollbar", background=CARD, troughcolor=BG,
                    borderwidth=0, arrowsize=10, gripcount=0)
    # ---------------- SERVER TAB ----------------
    def build_server(self):
        f = tk.Frame(self.tab_srv, bg=BG)
        f.pack(fill="both", expand=True, padx=25, pady=20)
        self.bin  = tk.StringVar(value=self.config.get("server_binary", "./build/bin/llama-server"))
        self.model= tk.StringVar(value=self.config.get("model_file", ""))
        self.proj = tk.StringVar(value=self.config.get("projector_file", ""))
        self.port = tk.StringVar(value=self.config.get("port", DEFAULT_PORT))
        self.ctx  = tk.StringVar(value=self.config.get("context", DEFAULT_CTX))
        self.gpu  = tk.StringVar(value=self.config.get("gpu_layers", DEFAULT_GPU))
        for var, key in [(self.bin, "server_binary"), (self.model, "model_file"),
                         (self.proj, "projector_file"), (self.port, "port"),
                         (self.ctx, "context"), (self.gpu, "gpu_layers")]:
            var.trace_add("write", lambda *_, v=var, k=key: self.config.set(k, v.get()))
        self.detect_binary()
        for label, var, browse in [("Server Binary", self.bin, True),
                                   ("Model (.gguf)", self.model, True),
                                   ("Projector (.gguf)", self.proj, True)]:
            self.field(f, label, var, browse)
        ttk.Frame(f, height=12).pack()
        params = tk.Frame(f, bg=BG)
        params.pack(fill="x")
        for lbl, v in [("Port", self.port), ("Context", self.ctx), ("GPU Layers", self.gpu)]:
            col = tk.Frame(params, bg=BG)
            col.pack(side="left", fill="x", expand=True, padx=3)
            tk.Label(col, text=lbl, bg=BG, fg=DIM, font=("Sans", 7)).pack(anchor="w", pady=(0, 2))
            tk.Entry(col, textvariable=v, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                     font=("Sans", 8), insertbackground=BLUE, justify="center").pack(fill="x", ipady=5)
        ttk.Frame(f, height=8).pack()
        vram_frame = tk.Frame(f, bg=BG)
        vram_frame.pack(fill="x")
        self.vram_label = tk.Label(vram_frame, text="Checking VRAM...", bg=BG, fg=DIM, font=("Sans", 7))
        self.vram_label.pack(side="left", fill="x", expand=True)
        self.btn_kill_gpu = self.btn(vram_frame, "Kill GPU Processes", RED, self.kill_gpu_processes)
        self.btn_kill_gpu.pack(side="right")
        ttk.Frame(f, height=4).pack()
        tip = tk.Frame(f, bg=CARD)
        tip.pack(fill="x", padx=1, pady=1)
        tk.Label(tip, text="16GB VRAM defaults: Context 8192, GPU Layers 99, Batch 512",
                 bg=CARD, fg=DIM, font=("Sans", 7)).pack(pady=5)
        ttk.Frame(f, height=12).pack()
        btns = tk.Frame(f, bg=BG)
        btns.pack(fill="x")
        self.btn_start = self.btn(btns, "Start Server", GREEN, self.start_server)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.btn_stop = self.btn(btns, "Stop Server", RED, self.stop_server)
        self.btn_stop.pack(side="left", fill="x", expand=True)
        self.btn_stop.config(state="disabled", bg=CARD)
        ttk.Frame(f, height=12).pack()
        log_frame = tk.Frame(f, bg=BG)
        log_frame.pack(fill="both", expand=True)
        self.log = scrolledtext.ScrolledText(log_frame, height=11, bg="#1a1d23", fg="#00ff00",
                                             bd=0, relief="flat", font=("Monospace", 7), wrap="word")
        self.log.pack(fill="both", expand=True)
    # ---------------- BATCH TAB ----------------
    def build_batch(self):
        main = tk.Frame(self.tab_batch, bg=BG)
        main.pack(fill="both", expand=True, padx=25, pady=15)
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        tool = tk.Frame(left, bg=BG)
        tool.pack(fill="x", pady=(0, 10))
        self.btn(tool, "Add Folder", BLUE, self.add_folder).pack(side="left", padx=(0, 8))
        self.btn_proc = self.btn(tool, "Start Processing", GREEN, self.toggle_batch)
        self.btn_proc.pack(side="left")
        self.overwrite = tk.BooleanVar(value=False)
        tk.Checkbutton(tool, text="Overwrite", variable=self.overwrite, bg=BG, fg=TEXT,
                       selectcolor=INPUT, activebackground=BG, font=("Sans", 8),
                       highlightthickness=0).pack(side="right")
        self.queue_scroll = ScrollFrame(left)
        self.queue_scroll.pack(fill="both", expand=True)
        prog = tk.Frame(left, bg=BG)
        prog.pack(fill="x", side="bottom", pady=(10, 0))
        self.progress = ttk.Progressbar(prog, mode="determinate")
        self.progress.pack(fill="x")
        self.prog_lbl = tk.Label(prog, text="Idle", bg=BG, fg=DIM, font=("Sans", 8))
        self.prog_lbl.pack(pady=(4, 0))
        right = tk.Frame(main, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(15, 0))
        tk.Label(right, text="Processing Status", bg=BG, fg=TEXT, font=("Sans", 9)).pack(anchor="w", pady=(0, 5))
        status_frame = tk.Frame(right, bg=BG)
        status_frame.pack(fill="both", expand=True)
        self.status_log = scrolledtext.ScrolledText(status_frame, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                                                    font=("Monospace", 7), wrap="word", state="disabled")
        self.status_log.pack(fill="both", expand=True)
    # ---------------- EDITOR TAB ----------------
    def build_editor(self):
        tool = tk.Frame(self.tab_editor, bg=BG)
        tool.pack(fill="x", padx=25, pady=15)
        self.btn(tool, "Load Folder", BLUE, self.load_editor_folder).pack(side="left")
        self.editor_folder_label = tk.Label(tool, text="No folder loaded", bg=BG, fg=DIM, font=("Sans", 8))
        self.editor_folder_label.pack(side="left", padx=15)
        content = tk.Frame(self.tab_editor, bg=BG)
        content.pack(fill="both", expand=True, padx=25, pady=(0, 15))
        left = tk.Frame(content, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="Images", bg=BG, fg=TEXT, font=("Sans", 8)).pack(anchor="w", pady=(0, 5))
        img_frame = tk.Frame(left, bg=BG)
        img_frame.pack(fill="both", expand=True)
        self.img_canvas = tk.Canvas(img_frame, bg=INPUT, highlightthickness=0, bd=0)
        img_scroll = ttk.Scrollbar(img_frame, orient="vertical", command=self.img_canvas.yview)
        self.img_list_frame = tk.Frame(self.img_canvas, bg=INPUT)
        self.img_list_frame.bind("<Configure>", lambda e: self.img_canvas.configure(scrollregion=self.img_canvas.bbox("all")))
        self.img_canvas.create_window((0, 0), window=self.img_list_frame, anchor="nw")
        self.img_canvas.configure(yscrollcommand=img_scroll.set)
        self.img_canvas.pack(side="left", fill="both", expand=True)
        img_scroll.pack(side="right", fill="y")
        right = tk.Frame(content, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(15, 0))
        tk.Label(right, text="Caption", bg=BG, fg=TEXT, font=("Sans", 8)).pack(anchor="w", pady=(0, 5))
        text_frame = tk.Frame(right, bg=BG)
        text_frame.pack(fill="both", expand=True)
        self.editor_text = scrolledtext.ScrolledText(text_frame, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                                                     font=("Sans", 9), wrap="word", insertbackground=BLUE)
        self.editor_text.pack(fill="both", expand=True)
        self.editor_text.bind("<KeyRelease>", self.autosave_caption)
        self.root.bind("<Up>", lambda e: self.editor_select_delta(-1))
        self.root.bind("<Down>", lambda e: self.editor_select_delta(1))
    # ---------------- CROP TAB ----------------
    def build_crop(self):
        f = tk.Frame(self.tab_crop, bg=BG)
        f.pack(fill="both", expand=True, padx=25, pady=20)
        
        self.crop_in = tk.StringVar()
        self.crop_out = tk.StringVar()
        self.crop_res = tk.StringVar(value="Auto (Best Fit)")
        self.crop_model = None
        self.crop_worker = None

        tk.Label(f, text="Input Folder:", bg=BG, fg=DIM, font=("Sans", 9)).pack(anchor="w")
        row_in = tk.Frame(f, bg=BG)
        row_in.pack(fill="x", pady=(0, 10))
        tk.Entry(row_in, textvariable=self.crop_in, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                 font=("Sans", 9), insertbackground=BLUE).pack(side="left", fill="x", expand=True, ipady=6)
        tk.Button(row_in, text="…", bg=CARD, fg=TEXT, bd=0, relief="flat", width=4,
                  command=self.crop_select_in).pack(side="right", padx=(4, 0))

        tk.Label(f, text="Output Folder:", bg=BG, fg=DIM, font=("Sans", 9)).pack(anchor="w")
        row_out = tk.Frame(f, bg=BG)
        row_out.pack(fill="x", pady=(0, 10))
        tk.Entry(row_out, textvariable=self.crop_out, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                 font=("Sans", 9), insertbackground=BLUE).pack(side="left", fill="x", expand=True, ipady=6)
        tk.Button(row_out, text="…", bg=CARD, fg=TEXT, bd=0, relief="flat", width=4,
                  command=lambda: self.crop_out.set(self._folder_picker("Select Output Folder"))).pack(side="right", padx=(4, 0))

        tk.Label(f, text="Resolution Strategy (Longest Side):", bg=BG, fg=DIM, font=("Sans", 9)).pack(anchor="w", pady=(0, 2))
        res_opts = ["Auto (Best Fit)", "768px", "1024px", "1536px", "2048px", "Keep Original (No Crop)"]
        self.crop_res_combo = ttk.Combobox(f, textvariable=self.crop_res, values=res_opts, state="readonly")
        self.crop_res_combo.pack(fill="x", ipady=4)
        
        tk.Label(f, text="Available Sizes: 768, 1024, 1536, 2048", bg=BG, fg=DIM, font=("Sans", 7)).pack(anchor="w", pady=(2, 0))
        
        tk.Frame(f, height=15, bg=BG).pack()
        self.btn_start_crop = self.btn(f, "START CROP PROCESSING", GREEN, self.start_crop)
        
        tk.Frame(f, height=10, bg=BG).pack()
        self.crop_progress = ttk.Progressbar(f, mode="determinate")
        self.crop_progress.pack(fill="x")
        
        self.crop_log_lbl = tk.Label(f, text="Ready.", bg=BG, fg=DIM, font=("Sans", 8))
        self.crop_log_lbl.pack(pady=(4, 0))

    def crop_select_in(self):
        path = self._folder_picker("Select Input Folder")
        if path:
            self.crop_in.set(path)
            if not self.crop_out.get():
                self.crop_out.set(str(Path(path) / "cropped_humans"))

    def start_crop(self):
        if not YOLO:
            messagebox.showerror("Error", "Ultralytics (YOLO) not installed. Please run 'pip install ultralytics'.")
            return
        
        input_path = self.crop_in.get()
        if not os.path.isdir(input_path):
            messagebox.showwarning("Error", "Input folder does not exist.")
            return

        self.crop_log_lbl.config(text="Loading Model... Please wait.")
        self.root.update_idletasks()
        
        try:
            if self.crop_model is None:
                self.crop_model = YOLO("yolov8n-seg.pt")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        res_str = self.crop_res.get()
        if res_str == "Keep Original (No Crop)":
            target = "KEEP"
        elif res_str == "Auto (Best Fit)":
            target = "AUTO"
        else:
            try:
                target = int(res_str.replace("px", ""))
            except:
                target = "AUTO"

        self.btn_start_crop.config(state="disabled", bg=CARD, text="Processing...")
        
        self.crop_worker = CropWorker(
            input_path,
            self.crop_out.get(),
            target,
            self.crop_model,
            self.update_crop_progress,
            self.update_crop_log,
            self.crop_finished
        )
        threading.Thread(target=self.crop_worker.run, daemon=True).start()

    def update_crop_progress(self, current, total):
        self.root.after(0, lambda: self.crop_progress.configure(maximum=total, value=current))

    def update_crop_log(self, msg):
        self.root.after(0, lambda: self.crop_log_lbl.config(text=msg))

    def crop_finished(self):
        self.root.after(0, self._crop_reset_ui)

    def _crop_reset_ui(self):
        self.btn_start_crop.config(state="normal", bg=GREEN, text="START CROP PROCESSING")
        self.crop_log_lbl.config(text="Finished.")

    # ---------------- FILTER & MOVE TAB ----------------
    def build_filter(self):
        f = tk.Frame(self.tab_filter, bg=BG)
        f.pack(fill="both", expand=True, padx=25, pady=20)
        # source
        tk.Label(f, text="Image-Caption folder:", bg=BG, fg=DIM, font=("Sans", 9)).pack(anchor="w")
        self.filter_src_var = tk.StringVar()
        self.field(f, "", self.filter_src_var, True)
        # keyword
        tk.Frame(f, height=8, bg=BG).pack()
        tk.Label(f, text="Keyword (case-insensitive):", bg=BG, fg=DIM, font=("Sans", 9)).pack(anchor="w")
        self.filter_kw_var = tk.StringVar()
        kw_entry = tk.Entry(f, textvariable=self.filter_kw_var, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                            font=("Sans", 10), insertbackground=BLUE)
        kw_entry.pack(fill="x", ipady=6)
        # target
        tk.Frame(f, height=8, bg=BG).pack()
        tk.Label(f, text="Target folder:", bg=BG, fg=DIM, font=("Sans", 9)).pack(anchor="w")
        self.filter_tgt_var = tk.StringVar()
        self.field(f, "", self.filter_tgt_var, True)
        # button
        tk.Frame(f, height=15, bg=BG).pack()
        self.btn(f, "Move matched pairs", BLUE, self.move_keyword_pairs).pack(anchor="e")
        # log
        tk.Frame(f, height=15, bg=BG).pack()
        log_frame = tk.Frame(f, bg=BG)
        log_frame.pack(fill="both", expand=True)
        self.filter_log = scrolledtext.ScrolledText(log_frame, height=10, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                                                    font=("Monospace", 8), wrap="word", state="disabled")
        self.filter_log.pack(fill="both", expand=True)
    def move_keyword_pairs(self):
        src = Path(self.filter_src_var.get())
        kw  = self.filter_kw_var.get().strip().lower()
        tgt = Path(self.filter_tgt_var.get())
        if not (src.is_dir() and tgt.is_dir() and kw):
            messagebox.showwarning("Input needed", "Please fill / validate all fields.")
            return
        # disable button & show progress
        self.filter_log.config(state="normal")
        self.filter_log.delete("1.0", "end")
        self.filter_log.insert("end", f"Searching for keyword: {kw}\n")
        self.filter_log.config(state="disabled")
        threading.Thread(target=self._move_worker, args=(src, kw, tgt), daemon=True).start()

    def _move_worker(self, src: Path, kw: str, tgt: Path):
        matched = 0
        try:
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                for img in sorted(src.glob(ext)):
                    txt = img.with_suffix(".txt")
                    if not txt.is_file():
                        continue
                    try:
                        content = txt.read_text(encoding="utf-8").lower()
                        if kw in content:
                            shutil.move(str(img), str(tgt / img.name))
                            shutil.move(str(txt), str(tgt / txt.name))
                            matched += 1
                            self.root.after(0, lambda m=img.name: self._log_filter(f"moved: {m}"))
                    except Exception as e:
                        self.root.after(0, lambda m=img.name, err=str(e): self._log_filter(f"error on {m}: {err}"))
        finally:
            self.root.after(0, lambda: self._log_filter(f"Done. Moved {matched} pairs."))

    def _log_filter(self, msg):
        self.filter_log.config(state="normal")
        self.filter_log.insert("end", msg + "\n")
        self.filter_log.see("end")
        self.filter_log.config(state="disabled")
    # ---------------- EDITOR UTILS ----------------
    def editor_select_delta(self, delta):
        if not self.editor_items:
            return
        idx = next((i for i, (_, f) in enumerate(self.editor_items) if f["bg"] == INPUT), 0)
        new = max(0, min(len(self.editor_items) - 1, idx + delta))
        self.editor_items[new][1].event_generate("<Button-1>")
    def load_editor_folder(self):
        path = Path(self._folder_picker())
        if path and path.is_dir():
            self.current_editor_folder = path
            self.editor_folder_label.config(text=path.name)
            self.load_editor_images()
    def create_editor_item(self, img_path: Path):
        item_frame = tk.Frame(self.img_list_frame, bg=CARD, cursor="hand2")
        item_frame.pack(fill="x", pady=2, padx=2)
        try:
            img = Image.open(img_path)
            img.thumbnail((self.thumb_size, self.thumb_size))
            photo = ImageTk.PhotoImage(img)
            img_label = tk.Label(item_frame, image=photo, bg=CARD)
            img_label.image = photo
            img_label.pack(side="left", padx=5, pady=5)
        except:
            img_label = tk.Label(item_frame, text="[img]", bg=CARD, fg=DIM, width=10)
            img_label.pack(side="left", padx=5, pady=5)
        name_label = tk.Label(item_frame, text=img_path.name,
                              bg=CARD, fg=TEXT, font=("Sans", 8), anchor="w")
        name_label.pack(side="left", fill="x", expand=True, padx=5)
        def select():
            self.load_caption_for_image(img_path)
            for w in self.img_list_frame.winfo_children():
                w.config(bg=CARD)
                for child in w.winfo_children():
                    child.config(bg=CARD)
            item_frame.config(bg=INPUT)
            for child in item_frame.winfo_children():
                if isinstance(child, tk.Label):
                    child.config(bg=INPUT)
        item_frame.bind("<Button-1>", lambda e: select())
        for child in item_frame.winfo_children():
            child.bind("<Button-1>", lambda e: select())
        img_label.bind("<Double-Button-1>", lambda e: self.show_zoom(img_path))
        name_label.bind("<Double-Button-1>", lambda e: self.show_zoom(img_path))
        self.editor_items.append((img_path, item_frame))
    def show_zoom(self, img_path: Path):
        if hasattr(self, "zoom_tl"):
            self.zoom_tl.destroy()
        tl = tk.Toplevel(self.root)
        tl.title(f"Zoom – {img_path.name}")
        tl.configure(bg=BG)
        tl.transient(self.root)
        x = self.root.winfo_x() - 550
        y = self.root.winfo_y() + 100
        tl.geometry(f"500x500+{x}+{y}")
        tl.focus()
        tl.bind("<Escape>", lambda e: tl.destroy())
        img = Image.open(img_path)
        img.thumbnail((480, 480), Image.LANCZOS)
        ph = ImageTk.PhotoImage(img)
        lbl = tk.Label(tl, image=ph, bg=BG)
        lbl.image = ph
        lbl.pack(expand=True)
        close = tk.Label(tl, text="✖  Close", fg=DIM, bg=BG, cursor="hand2")
        close.pack(pady=4)
        close.bind("<Button-1>", lambda e: tl.destroy())
        self.zoom_tl = tl
    def autosave_caption(self, event=None):
        if hasattr(self, "current_caption_file"):
            try:
                self.current_caption_file.write_text(self.editor_text.get("1.0", "end-1c"), encoding="utf-8")
            except Exception as e:
                print("Autosave error:", e)
    # ---------------- SERVER  ----------------
    def detect_binary(self):
        for p in ["./build/bin/llama-server", "./llama-server", "../llama.cpp/build/bin/llama-server"]:
            if Path(p).exists():
                self.bin.set(p)
                break
    def update_vram_info(self):
        try:
            used, total = map(int, subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip().split(","))
            free, percent = total - used, (used / total) * 100
            color = GREEN if percent < 50 else (BLUE if percent < 80 else RED)
            self.vram_label.config(text=f"VRAM: {used}MB used / {free}MB free / {total}MB total ({percent:.0f}%)", fg=color)
        except:
            self.vram_label.config(text="VRAM info unavailable", fg=DIM)
    def kill_gpu_processes(self):
        if not messagebox.askyesno("Kill GPU Processes", "Terminate ALL GPU processes?"):
            return
        try:
            pids = map(int, subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL).decode().strip().split())
            killed = 0
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed += 1
                except:
                    pass
            threading.Timer(1.0, self.update_vram_info).start()
            messagebox.showinfo("GPU Processes", f"Killed {killed} process(es).")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to kill GPU processes:\n{e}")
    # ---------------------------------------------------------
    #  Native folder picker using zenity for XFCE/GTK integration
    # ---------------------------------------------------------
    def _folder_picker(self, title="Select folder"):
        """Return POSIX path string; empty if cancelled. Uses zenity for native XFCE/GTK dialog."""
        # 1. Try zenity (GTK native)
        try:
            # check if zenity exists
            subprocess.run(["zenity", "--version"], capture_output=True, check=True)
            
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory", "--title", title],
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on cancel (exit code 1)
                timeout=60
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                if path: return path
            # If it returned 1, user cancelled. Return empty string.
            if result.returncode == 1:
                return ""
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        # 2. Fallback to custom Tk picker ONLY if zenity failed to run or wasn't found
        return self._folder_picker_tk(title)
    
    def _folder_picker_tk(self, title="Select folder"):
        """Custom Tk folder picker as fallback."""
        out = []
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry("600x400")
        top.configure(bg=BG)
        top.transient(self.root)
        top.grab_set()
        top.focus()

        current = Path.cwd()
        addr = tk.StringVar(value=str(current))

        # address bar
        bar = tk.Frame(top, bg=BG)
        bar.pack(fill="x", padx=6, pady=6)
        tk.Entry(bar, textvariable=addr, bg=INPUT, fg=TEXT, bd=0, font=("Sans", 9)).pack(side="left", fill="x", expand=True, ipady=5)

        # file list
        frame = tk.Frame(top, bg=BG)
        frame.pack(fill="both", expand=True)
        lb_frame = tk.Frame(frame, bg=BG)
        lb_frame.pack(fill="both", expand=True, padx=6, pady=6)
        lb = tk.Listbox(lb_frame, bg=INPUT, fg=TEXT, bd=0, selectbackground=BLUE,
                        selectforeground="white", font=("Sans", 9))
        lb.pack(side="left", fill="both", expand=True)
        sc = ttk.Scrollbar(lb_frame, orient="vertical", command=lb.yview)
        sc.pack(side="right", fill="y")
        lb.configure(yscrollcommand=sc.set)

        # buttons
        btn_bar = tk.Frame(top, bg=BG)
        btn_bar.pack(fill="x", padx=6, pady=6)
        tk.Button(btn_bar, text="Select", bg=GREEN, fg="white", bd=0, relief="flat",
                  command=lambda: _select()).pack(side="right", padx=(6, 0))
        tk.Button(btn_bar, text="Cancel", bg=CARD, fg=TEXT, bd=0, relief="flat",
                  command=lambda: _cancel()).pack(side="right")

        def _populate():
            lb.delete(0, tk.END)
            p = Path(addr.get())
            if p.parent:
                lb.insert(0, "..")
            for d in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                lb.insert(tk.END, d.name + ("/" if d.is_dir() else ""))

        def _select():
            sel = lb.curselection()
            if not sel:
                out.append(addr.get())
                top.destroy()
                return
            name = lb.get(sel[0])
            if name == "..":
                addr.set(str(Path(addr.get()).parent))
            else:
                new = Path(addr.get()) / name.rstrip("/")
                if new.is_dir():
                    addr.set(str(new))
                else:
                    out.append(str(new))
                    top.destroy()
                    return
            _populate()

        def _cancel():
            top.destroy()

        lb.bind("<Double-Button-1>", lambda e: _select())
        _populate()
        self.root.wait_window(top)
        return out[0] if out else ""

    # ---------------- SERVER CONTROL (NO-HANG) ----------------
    def start_server(self):
        cmd = [self.bin.get(), "-m", self.model.get(), "--port", self.port.get(),
               "--ctx-size", self.ctx.get(), "-ngl", self.gpu.get(), "-b", DEFAULT_BATCH]
        if self.proj.get():
            cmd.extend(["--mmproj", self.proj.get()])
        self.log.insert("end", "Starting server...\n")
        try:
            self.server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True,
                                               bufsize=1, preexec_fn=os.setsid)
            self.btn_start.config(state="disabled", bg=CARD)
            self.btn_stop.config(state="normal", bg=RED)
            threading.Thread(target=self.watch_server, daemon=True).start()
        except Exception as e:
            self.log.insert("end", f"Error: {e}\n")
    def stop_server(self):
        if self.server_proc:
            try:
                os.killpg(os.getpgid(self.server_proc.pid), signal.SIGTERM)
                threading.Thread(target=self.server_proc.wait, daemon=True).start()
            except Exception as e:
                self.log.insert("end", f"Stop error: {e}\n")
        self.root.after(100, self.reset_ui)
    def watch_server(self):
        try:
            for line in iter(self.server_proc.stdout.readline, ""):
                if line:
                    self.log.insert("end", line)
                    self.log.see("end")
        except Exception:
            pass
        self.root.after(0, self.reset_ui)
    def reset_ui(self):
        self.btn_start.config(state="normal", bg=GREEN)
        self.btn_stop.config(state="disabled", bg=CARD)
        self.log.insert("end", "Server stopped\n")
    # ---------------- BATCH  (USES CUSTOM PROMPT PER FOLDER) ----------------
    def add_folder(self):
        path = Path(self._folder_picker("Select folder"))
        if path and path.is_dir():
            item = QueueItem(self.queue_scroll.content, path, self.remove_item, self.config)
            item.pack(fill="x", pady=(0, 6))
            self.queue.append(item)
    def remove_item(self, item):
        if item.status != "processing":
            item.destroy()
            if item in self.queue:
                self.queue.remove(item)
    def log_status(self, msg):
        self.status_log.config(state="normal")
        self.status_log.insert("end", f"{msg}\n")
        self.status_log.see("end")
        self.status_log.config(state="disabled")
    def toggle_batch(self):
        if self.batch_running:
            self.batch_running = False
            self.btn_proc.config(text="Start Processing", bg=GREEN)
            self.prog_lbl.config(text="Stopping...")
        else:
            try:
                self.client = OpenAI(base_url=API_URL, api_key="sk-no-key")
                self.client.models.list()
            except Exception as e:
                messagebox.showerror("Connection Error", f"Cannot connect to server.\n{e}")
                return
            self.status_log.config(state="normal")
            self.status_log.delete("1.0", "end")
            self.status_log.config(state="disabled")
            self.batch_running = True
            self.btn_proc.config(text="Stop Processing", bg=RED)
            threading.Thread(target=self.run_batch, daemon=True).start()
    def run_batch(self):
        total = len(self.queue)
        if total == 0:
            self.batch_running = False
            self.root.after(0, lambda: self.btn_proc.config(text="Start Processing", bg=GREEN))
            self.root.after(0, lambda: self.prog_lbl.config(text="No folders in queue"))
            return
        for idx, item in enumerate(self.queue):
            if not self.batch_running:
                break
            self.root.after(0, lambda i=item: i.set_status("processing", "Scanning..."))
            imgs = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                imgs.extend(sorted(Path(item.folder_path).glob(ext)))
            total_imgs = len(imgs)
            done = 0
            for img in imgs:
                if not self.batch_running:
                    break
                txt = img.with_suffix(".txt")
                if txt.exists() and not self.overwrite.get():
                    done += 1
                    continue
                try:
                    #  ➜➜➜  USE THE PROMPT WRITTEN IN THE GUI FOR THIS FOLDER  ➜➜➜
                    prompt = item.get_prompt() or DEFAULT_PROMPT
                    b64 = base64.b64encode(img.read_bytes()).decode()
                    resp = self.client.chat.completions.create(
                        model=Path(self.model.get()).stem,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]}],
                        max_tokens=300
                    )
                    txt.write_text(resp.choices[0].message.content.strip(), encoding="utf-8")
                    done += 1
                    self.root.after(0, lambda name=img.name: self.log_status(f"✓ {name}"))
                except Exception as e:
                    self.root.after(0, lambda name=img.name, err=str(e): self.log_status(f"✗ {name}: {err}"))
                pct = int(((idx + (done / total_imgs)) / total) * 100) if total_imgs > 0 else 0
                self.root.after(0, lambda p=pct: self.progress.configure(value=p))
                self.root.after(0, lambda d=done, t=total_imgs, i=item:
                               i.set_status("processing", f"{d}/{t}"))
            self.root.after(0, lambda i=item, r=self.batch_running:
                           i.set_status("done" if r else "error", "Complete" if r else "Stopped"))
        self.batch_running = False
        self.root.after(0, lambda: self.btn_proc.config(text="Start Processing", bg=GREEN))
        self.root.after(0, lambda: self.prog_lbl.config(text="Idle"))
    # ---------------- MISSING METHODS ----------------
    def load_editor_images(self):
        """Load all images from the current editor folder."""
        # Clear existing items
        for widget in self.img_list_frame.winfo_children():
            widget.destroy()
        self.editor_items = []
        
        if not self.current_editor_folder:
            return
        
        # Load images
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            for img_path in sorted(self.current_editor_folder.glob(ext)):
                self.create_editor_item(img_path)
        
        # Select first image if available
        if self.editor_items:
            self.editor_items[0][1].event_generate("<Button-1>")
    
    def load_caption_for_image(self, img_path: Path):
        """Load the caption text file for the selected image."""
        self.current_caption_file = img_path.with_suffix(".txt")
        try:
            if self.current_caption_file.exists():
                content = self.current_caption_file.read_text(encoding="utf-8")
                self.editor_text.delete("1.0", "end")
                self.editor_text.insert("1.0", content)
            else:
                self.editor_text.delete("1.0", "end")
        except Exception as e:
            print(f"Error loading caption: {e}")
    
    def field(self, parent, label, var, browse):
        """Create an input field with optional browse button."""
        if label:
            tk.Label(parent, text=label, bg=BG, fg=DIM, font=("Sans", 8)).pack(anchor="w", pady=(0, 2))
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(0, 10))
        tk.Entry(row, textvariable=var, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                 font=("Sans", 9), insertbackground=BLUE).pack(side="left", fill="x", expand=True, ipady=6)
        if browse:
            tk.Button(row, text="…", bg=CARD, fg=TEXT, bd=0, relief="flat",
                      width=4, command=lambda: var.set(self._folder_picker(f"Select {label}"))).pack(side="right", padx=(4, 0))
    
    def btn(self, parent, text, color, cmd):
        """Create a styled button."""
        btn = tk.Button(parent, text=text, bg=color, fg="white", bd=0, relief="flat",
                        font=("Sans", 9), cursor="hand2", command=cmd)
        btn.pack(fill="x", ipady=6)
        return btn
    # ---------------- CLEAN EXIT ----------------
    def on_close(self):
        if self.server_proc:
            self.stop_server()
        self.root.destroy()

# ---------------- RUN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
