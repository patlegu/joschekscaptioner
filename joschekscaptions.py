#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joschek’s Captioner  v22
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

# ---------------- MODERN PALETTE ----------------
# Primary colors - sleek dark theme with modern accents
BG = "#1a1b26"              # Darker background for better contrast
CARD = "#24283b"            # Modern card color
INPUT = "#2a2d3e"           # Input field background
TEXT = "#f8f8f2"            # Brighter text for better readability
DIM = "#6272a4"             # Subtle dim text
BORDER = "#44475a"          # Border color
ACCENT = "#6272a4"          # Primary accent color
BLUE = "#5294e2"            # Modern blue
GREEN = "#50fa7b"           # Vibrant green
RED = "#ff5555"             # Modern red
PURPLE = "#bd93f9"          # Purple accent
CYAN = "#8be9fd"            # Cyan accent
ORANGE = "#ff79c6"          # Orange accent
HOVER = "#3d424e"           # Hover state color

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
                              highlightthickness=0, font=("Sans", 8), insertbackground=BLUE, wrap="word")
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
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview,
                                 width=30, bg=CARD, troughcolor=BG, bd=0, highlightthickness=0,
                                 activebackground=HOVER, elementborderwidth=0)
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
        root.title("Joschek's Captioner V 1.0")
        root.geometry("1100x720")
        root.configure(bg=BG)
        # Set modern font for the entire application
        root.option_add("*Font", ("Sans", 10))
        # Set window icon if available (Linux)
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "jc32x32.png")
            if os.path.exists(icon_path):
                self.icon_img = tk.PhotoImage(file=icon_path)
                root.iconphoto(True, self.icon_img)
        except Exception as e:
            print(f"Error loading icon: {e}")

        self.config = Config()
        self.setup_styles()
        self.server_proc = None
        self.batch_running = False
        self.queue = []
        self.client = None
        self.current_editor_folder = None
        self.editor_items = []
        self.thumb_size = 200 # Fixed height for editor images
        self.thumb_cache = {}
        # Create modern custom tab system
        self.create_modern_tab_system()
        self.build_server()
        self.build_batch()
        self.build_editor()
        self.build_filter()
        self.build_crop()
        self.update_vram_info() # Start VRAM monitoring
        root.protocol("WM_DELETE_WINDOW", self.on_close)
    # ---------------- MODERN STYLES ----------------
    def setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")

        # Modern progress bar
        s.configure("TProgressbar", background=BLUE, troughcolor=BG, borderwidth=0, thickness=6)
        s.configure("Horizontal.TProgressbar", background=BLUE, troughcolor=BG, borderwidth=0, thickness=6)

        # Modern subtle scrollbar - wider for better accessibility
        s.configure("Vertical.TScrollbar", background=BG, troughcolor=BG, borderwidth=0, arrowsize=0, width=30)
        s.map("Vertical.TScrollbar", background=[("active", HOVER), ("!active", CARD)])
        
        s.configure("Horizontal.TScrollbar", background=BG, troughcolor=BG, borderwidth=0, arrowsize=0, width=30)
        s.map("Horizontal.TScrollbar", background=[("active", HOVER), ("!active", CARD)])

        # Modern combobox
        s.configure("TCombobox", background=INPUT, foreground=TEXT, borderwidth=0,
                    fieldbackground=INPUT, selectbackground=BLUE, selectforeground=TEXT,
                    arrowcolor=TEXT)
        s.map("TCombobox", fieldbackground=[("readonly", INPUT)], 
              background=[("readonly", INPUT)], 
              foreground=[("readonly", TEXT)])

    # ---------------- CUSTOM TAB SYSTEM ----------------
    def create_modern_tab_system(self):
        # Main container
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        # Top Bar (Tabs)
        self.tab_bar = tk.Frame(main, bg=BG, height=50)
        self.tab_bar.pack(side="top", fill="x", pady=(10, 0), padx=20)
        
        # Content Area
        self.content_area = tk.Frame(main, bg=BG)
        self.content_area.pack(side="bottom", fill="both", expand=True)
        
        # Initialize containers for each tab's content
        self.tab_srv = tk.Frame(self.content_area, bg=BG)
        self.tab_batch = tk.Frame(self.content_area, bg=BG)
        self.tab_editor = tk.Frame(self.content_area, bg=BG)
        self.tab_filter = tk.Frame(self.content_area, bg=BG)
        self.tab_crop = tk.Frame(self.content_area, bg=BG)
        
        self.tabs = {}        # Name -> Frame
        self.tab_btns = {}    # Name -> Label widget
        self.current_tab = None

        # Add tabs
        self.add_tab("Server", self.tab_srv)
        self.add_tab("Batch", self.tab_batch)
        self.add_tab("Editor", self.tab_editor)
        self.add_tab("Filter & Move", self.tab_filter)
        self.add_tab("Crop", self.tab_crop)

        # Separator line under tabs
        tk.Frame(main, bg=INPUT, height=1).pack(side="top", fill="x", pady=(0, 0))

        # Select first
        self.switch_tab("Server")

    def add_tab(self, name, frame):
        self.tabs[name] = frame
        btn = tk.Label(self.tab_bar, text=name, bg=BG, fg=DIM, font=("Sans", 10, "bold"),
                       cursor="hand2", padx=15, pady=8)
        btn.pack(side="left")
        btn.bind("<Button-1>", lambda e: self.switch_tab(name))
        btn.bind("<Enter>", lambda e: self._hover_tab(name, True))
        btn.bind("<Leave>", lambda e: self._hover_tab(name, False))
        self.tab_btns[name] = btn

    def switch_tab(self, name):
        if self.current_tab == name:
            return
        
        # Hide old
        if self.current_tab:
            self.tabs[self.current_tab].pack_forget()
            self.tab_btns[self.current_tab].config(fg=DIM)
        
        # Show new
        self.current_tab = name
        self.tabs[name].pack(fill="both", expand=True)
        self.tab_btns[name].config(fg=BLUE) # Active color

    def _hover_tab(self, name, entering):
        if name == self.current_tab:
            return
        self.tab_btns[name].config(fg=TEXT if entering else DIM)

    # ---------------- SERVER TAB ----------------
    def build_server(self):
        f = tk.Frame(self.tab_srv, bg=BG)
        f.pack(fill="both", expand=True, padx=25, pady=20)
        
        default_bin = "./build/bin/llama-server"
        if os.name == 'nt':
            default_bin = "llama-server.exe"
            
        self.bin  = tk.StringVar(value=self.config.get("server_binary", default_bin))
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
        tk.Frame(f, height=12, bg=BG).pack()
        params = tk.Frame(f, bg=BG)
        params.pack(fill="x")
        for lbl, v in [("Port", self.port), ("Context", self.ctx), ("GPU Layers", self.gpu)]:
            col = tk.Frame(params, bg=BG)
            col.pack(side="left", fill="x", expand=True, padx=3)
            tk.Label(col, text=lbl, bg=BG, fg=DIM, font=("Sans", 7)).pack(anchor="w", pady=(0, 2))
            tk.Entry(col, textvariable=v, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                     highlightthickness=0, font=("Sans", 8), insertbackground=BLUE, justify="center").pack(fill="x", ipady=5)
        tk.Frame(f, height=8, bg=BG).pack()
        vram_frame = tk.Frame(f, bg=BG)
        vram_frame.pack(fill="x")
        self.vram_label = tk.Label(vram_frame, text="Checking VRAM...", bg=BG, fg=DIM, font=("Sans", 7))
        self.vram_label.pack(side="left", fill="x", expand=True)
        self.btn_kill_gpu = self.btn(vram_frame, "Kill GPU Processes", BLUE, self.kill_gpu_processes)
        self.btn_kill_gpu.pack(side="right")
        tk.Frame(f, height=4, bg=BG).pack()
        tip = tk.Frame(f, bg=CARD)
        tip.pack(fill="x", padx=1, pady=1)
        tk.Label(tip, text="16GB VRAM defaults: Context 8192, GPU Layers 99, Batch 512",
                 bg=CARD, fg=DIM, font=("Sans", 7)).pack(pady=5)
        tk.Frame(f, height=12, bg=BG).pack()
        btns = tk.Frame(f, bg=BG)
        btns.pack(fill="x")
        self.btn_start = self.btn(btns, "Start Server", BLUE, self.start_server)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.btn_stop = self.btn(btns, "Stop Server", BLUE, self.stop_server)
        self.btn_stop.pack(side="left", fill="x", expand=True)
        self.btn_stop.config(state="disabled", bg=CARD)
        tk.Frame(f, height=12, bg=BG).pack()
        log_frame = tk.Frame(f, bg=BG)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=11, bg="#1a1d23", fg="#00ff00",
                           bd=0, relief="flat", highlightthickness=0, font=("Monospace", 7), wrap="word")
        self.log.pack(side="left", fill="both", expand=True)
        s_log = tk.Scrollbar(log_frame, orient="vertical", command=self.log.yview,
                             width=30, bg=CARD, troughcolor=BG, bd=0, highlightthickness=0,
                             activebackground=HOVER, elementborderwidth=0)
        s_log.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=s_log.set)
    # ---------------- BATCH TAB ----------------
    def build_batch(self):
        main = tk.Frame(self.tab_batch, bg=BG)
        main.pack(fill="both", expand=True, padx=25, pady=15)
        # Left side takes most room
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        tool = tk.Frame(left, bg=BG)
        tool.pack(fill="x", pady=(0, 10))
        self.btn(tool, "Add Folder", BLUE, self.add_folder).pack(side="left", padx=(0, 8))
        self.btn_proc = self.btn(tool, "Start Processing", BLUE, self.toggle_batch)
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
        # Right side is thinner for status
        right = tk.Frame(main, bg=BG, width=250)
        right.pack(side="right", fill="both", expand=False, padx=(15, 0))
        right.pack_propagate(False)
        tk.Label(right, text="Processing Status", bg=BG, fg=TEXT, font=("Sans", 9)).pack(anchor="w", pady=(0, 5))
        status_frame = tk.Frame(right, bg=BG)
        status_frame.pack(fill="both", expand=True)
        self.status_log = tk.Text(status_frame, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                                  highlightthickness=0, font=("Monospace", 7), wrap="word", state="disabled")
        self.status_log.pack(side="left", fill="both", expand=True)
        s_batch = tk.Scrollbar(status_frame, orient="vertical", command=self.status_log.yview,
                               width=30, bg=CARD, troughcolor=BG, bd=0, highlightthickness=0,
                               activebackground=HOVER, elementborderwidth=0)
        s_batch.pack(side="right", fill="y")
        self.status_log.configure(yscrollcommand=s_batch.set)
    # ---------------- EDITOR TAB ----------------
    def build_editor(self):
        tool = tk.Frame(self.tab_editor, bg=BG)
        tool.pack(fill="x", padx=25, pady=15)
        self.btn(tool, "Load Folder", BLUE, self.load_editor_folder).pack(side="left")
        self.editor_folder_label = tk.Label(tool, text="No folder loaded", bg=BG, fg=DIM, font=("Sans", 8))
        self.editor_folder_label.pack(side="left", padx=15)

        # Add filter input
        filter_frame = tk.Frame(tool, bg=BG)
        filter_frame.pack(side="right")
        tk.Label(filter_frame, text="Filter (in captions):", bg=BG, fg=DIM, font=("Sans", 8)).pack(side="left", padx=(10, 2))
        self.editor_filter_var = tk.StringVar()
        filter_entry = tk.Entry(filter_frame, textvariable=self.editor_filter_var, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                              highlightthickness=0, font=("Sans", 8), insertbackground=BLUE, width=20)
        filter_entry.pack(side="left")
        filter_entry.bind("<Enter>", lambda e: filter_entry.config(bg=HOVER))
        filter_entry.bind("<Leave>", lambda e: filter_entry.config(bg=INPUT))
        filter_entry.bind("<Return>", lambda e: self.apply_editor_filter())
        clear_btn = tk.Button(filter_frame, text="Clear", bg=CARD, fg=TEXT, bd=0, relief="flat",
                            font=("Sans", 8), cursor="hand2", command=self.clear_editor_filter)
        clear_btn.pack(side="left", padx=(2, 0))
        clear_btn.bind("<Enter>", lambda e: clear_btn.config(bg=HOVER))
        clear_btn.bind("<Leave>", lambda e: clear_btn.config(bg=CARD))

        # Main scrollable area for items
        content = tk.Frame(self.tab_editor, bg=BG)
        content.pack(fill="both", expand=True, padx=25, pady=(0, 15))
        
        self.img_canvas = tk.Canvas(content, bg=BG, highlightthickness=0, bd=0)
        # Custom wide minimal scrollbar
        img_scroll = tk.Scrollbar(content, orient="vertical", command=self.img_canvas.yview,
                                  width=30, bg=CARD, troughcolor=BG, bd=0, highlightthickness=0,
                                  activebackground=HOVER, elementborderwidth=0)
        
        self.img_list_frame = tk.Frame(self.img_canvas, bg=BG)
        self.canvas_window = self.img_canvas.create_window((0, 0), window=self.img_list_frame, anchor="nw")
        
        def _on_canvas_configure(e):
            self.img_canvas.itemconfig(self.canvas_window, width=e.width)
        
        self.img_canvas.bind("<Configure>", _on_canvas_configure)
        self.img_list_frame.bind("<Configure>", lambda e: self.img_canvas.configure(scrollregion=self.img_canvas.bbox("all")))
        
        # Lazy loading hook
        def _on_scroll(*args):
            img_scroll.set(*args)
            if float(args[1]) > 0.9:
                self.load_more_items()
        
        self.img_canvas.configure(yscrollcommand=_on_scroll)
        self.img_canvas.pack(side="left", fill="both", expand=True)
        img_scroll.pack(side="right", fill="y")

        # Bind mousewheel to canvas with robust cross-platform support
        def _on_mousewheel(e):
            if e.num == 4 or e.delta > 0:
                self.img_canvas.yview_scroll(-2, "units")
            elif e.num == 5 or e.delta < 0:
                self.img_canvas.yview_scroll(2, "units")

        # Bind to canvas and all its children recursively when mouse enters
        def _bind_mousewheel(e):
            self.img_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.img_canvas.bind_all("<Button-4>", _on_mousewheel)
            self.img_canvas.bind_all("<Button-5>", _on_mousewheel)
        
        def _unbind_mousewheel(e):
            self.img_canvas.unbind_all("<MouseWheel>")
            self.img_canvas.unbind_all("<Button-4>")
            self.img_canvas.unbind_all("<Button-5>")

        self.img_canvas.bind("<Enter>", _bind_mousewheel)
        self.img_canvas.bind("<Leave>", _unbind_mousewheel)
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
                 highlightthickness=0, font=("Sans", 9), insertbackground=BLUE).pack(side="left", fill="x", expand=True, ipady=6)
        tk.Button(row_in, text="…", bg=CARD, fg=TEXT, bd=0, relief="flat", highlightthickness=0, width=4,
                  command=self.crop_select_in).pack(side="right", padx=(4, 0))

        tk.Label(f, text="Output Folder:", bg=BG, fg=DIM, font=("Sans", 9)).pack(anchor="w")
        row_out = tk.Frame(f, bg=BG)
        row_out.pack(fill="x", pady=(0, 10))
        tk.Entry(row_out, textvariable=self.crop_out, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                 highlightthickness=0, font=("Sans", 9), insertbackground=BLUE).pack(side="left", fill="x", expand=True, ipady=6)
        tk.Button(row_out, text="…", bg=CARD, fg=TEXT, bd=0, relief="flat", highlightthickness=0, width=4,
                  command=lambda: self.crop_out.set(self._folder_picker("Select Output Folder"))).pack(side="right", padx=(4, 0))

        tk.Label(f, text="Resolution Strategy (Longest Side):", bg=BG, fg=DIM, font=("Sans", 9)).pack(anchor="w", pady=(0, 2))
        res_opts = ["Auto (Best Fit)", "768px", "1024px", "1536px", "2048px", "Keep Original (No Crop)"]
        
        # Custom dark minimalist dropdown
        self.res_mb = tk.Menubutton(f, textvariable=self.crop_res, bg=INPUT, fg=TEXT, 
                                    activebackground=HOVER, activeforeground=TEXT,
                                    bd=0, highlightthickness=0, font=("Sans", 9),
                                    relief="flat", anchor="w", padx=10)
        self.res_menu = tk.Menu(self.res_mb, tearoff=0, bg=CARD, fg=TEXT, 
                                activebackground=BLUE, activeforeground=TEXT, bd=0)
        self.res_mb["menu"] = self.res_menu
        for opt in res_opts:
            self.res_menu.add_command(label=opt, command=lambda o=opt: self.crop_res.set(o))
        self.res_mb.pack(fill="x", ipady=8)
        
        tk.Label(f, text="Available Sizes: 768, 1024, 1536, 2048", bg=BG, fg=DIM, font=("Sans", 7)).pack(anchor="w", pady=(2, 0))
        
        tk.Frame(f, height=15, bg=BG).pack()
        self.btn_start_crop = self.btn(f, "START CROP PROCESSING", BLUE, self.start_crop)
        
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
        self.btn_start_crop.config(state="normal", bg=BLUE, text="START CROP PROCESSING")
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
                            highlightthickness=0, font=("Sans", 10), insertbackground=BLUE)
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
        self.filter_log = tk.Text(log_frame, height=10, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                                  highlightthickness=0, font=("Monospace", 8), wrap="word", state="disabled")
        self.filter_log.pack(side="left", fill="both", expand=True)
        s_filter = tk.Scrollbar(log_frame, orient="vertical", command=self.filter_log.yview,
                                width=30, bg=CARD, troughcolor=BG, bd=0, highlightthickness=0,
                                activebackground=HOVER, elementborderwidth=0)
        s_filter.pack(side="right", fill="y")
        self.filter_log.configure(yscrollcommand=s_filter.set)
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
    def load_editor_folder(self):
        path = Path(self._folder_picker())
        if path and path.is_dir():
            self.current_editor_folder = path
            self.editor_folder_label.config(text=path.name)
            self.load_editor_images()

    def create_editor_item(self, img_path: Path, thumb_img=None):
        item_frame = tk.Frame(self.img_list_frame, bg=CARD, height=self.thumb_size + 20)
        item_frame.pack(fill="x", pady=5, padx=2)
        item_frame.pack_propagate(False)
        
        # Image Container
        img_container = tk.Frame(item_frame, bg=CARD, width=self.thumb_size + 20)
        img_container.pack(side="left", fill="y")
        
        # Placeholder or Image
        img_label = tk.Label(img_container, bg=CARD, fg=DIM)
        if thumb_img:
            photo = ImageTk.PhotoImage(thumb_img)
            img_label.configure(image=photo)
            img_label.image = photo
        else:
            img_label.configure(text="Loading...")
            
        img_label.pack(expand=True)
        img_label.bind("<Double-Button-1>", lambda e: self.show_zoom(img_path))
        
        # Info & Text Container
        right_container = tk.Frame(item_frame, bg=CARD)
        right_container.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        tk.Label(right_container, text=img_path.name, bg=CARD, fg=DIM, font=("Sans", 8), anchor="w").pack(fill="x")
        
        txt_path = img_path.with_suffix(".txt")
        initial_content = ""
        if txt_path.exists():
            try: initial_content = txt_path.read_text(encoding="utf-8", errors="ignore")
            except: pass
            
        text_area = tk.Text(right_container, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                            highlightthickness=0, font=("Sans", 9), wrap="word", insertbackground=BLUE)
        text_area.insert("1.0", initial_content)
        text_area.pack(fill="both", expand=True, pady=(5, 0))
        
        def _save(event=None):
            content = text_area.get("1.0", "end-1c")
            try: txt_path.write_text(content, encoding="utf-8")
            except: pass
            
        text_area.bind("<KeyRelease>", _save)
        
        self.editor_items.append((img_path, item_frame))
        return img_label
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
    # ---------------- SERVER  ----------------
    def detect_binary(self):
        paths = ["./build/bin/llama-server", "./llama-server", "../llama.cpp/build/bin/llama-server"]
        if os.name == 'nt':
            paths = ["llama-server.exe", "build/bin/Release/llama-server.exe", "./llama-server.exe"]
            
        for p in paths:
            if Path(p).exists():
                self.bin.set(p)
                break
    def update_vram_info(self):
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            if not output: raise ValueError("Empty output")
            used, total = map(int, output.split(","))
            free, percent = total - used, (used / total) * 100
            color = GREEN if percent < 50 else (BLUE if percent < 80 else RED)
            self.vram_label.config(text=f"VRAM: {used}MB used / {free}MB free / {total}MB total ({percent:.0f}%)", fg=color)
        except Exception as e:
            self.vram_label.config(text=f"VRAM info unavailable", fg=DIM)
        # Schedule next update
        self.root.after(5000, self.update_vram_info)
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
    #  Cross-platform folder picker
    # ---------------------------------------------------------
    def _folder_picker(self, title="Select folder"):
        """Return path string; empty if cancelled."""
        
        # Windows: Use native Tkinter dialog (it's actually native on Windows)
        if os.name == 'nt':
            return filedialog.askdirectory(title=title)
            
        # Linux/Unix: Try zenity for better DE integration (XFCE/GNOME/etc)
        try:
            # check if zenity exists
            subprocess.run(["zenity", "--version"], capture_output=True, check=True)
            
            # Use zenity without timeout as user interaction takes time
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory", "--title", title],
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on cancel (exit code 1)
            )
            
            if result.returncode == 0:
                path = result.stdout.strip()
                if path: return path
            # If user cancelled (code 1), return empty string immediately
            # do NOT fall back to tk picker
            return ""
            
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Zenity not found or failed to execute -> Fallback
            pass
        except Exception as e:
            print(f"Zenity error: {e}")
            pass

        # Fallback to custom Tk picker if zenity wasn't found
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
        tk.Entry(bar, textvariable=addr, bg=INPUT, fg=TEXT, bd=0, highlightthickness=0, font=("Sans", 9)).pack(side="left", fill="x", expand=True, ipady=5)

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
        
        # Windows compatibility for subprocess
        kwargs = {}
        if os.name != 'nt':
            kwargs['preexec_fn'] = os.setsid
        else:
            # CREATE_NEW_PROCESS_GROUP = 0x00000200
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            self.server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True,
                                               bufsize=1, **kwargs)
            self.btn_start.config(state="disabled", bg=CARD)
            self.btn_stop.config(state="normal", bg=RED)
            threading.Thread(target=self.watch_server, daemon=True).start()
        except Exception as e:
            self.log.insert("end", f"Error: {e}\n")

    def stop_server(self):
        if self.server_proc:
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.server_proc.pid), signal.SIGTERM)
                else:
                    # Windows: Send CTRL_BREAK_EVENT or force kill
                    self.server_proc.terminate() 
                
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
        self.btn_start.config(state="normal", bg=BLUE)
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
            self.btn_proc.config(text="Start Processing", bg=BLUE)
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
            self.root.after(0, lambda: self.btn_proc.config(text="Start Processing", bg=BLUE))
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
        self.root.after(0, lambda: self.btn_proc.config(text="Start Processing", bg=BLUE))
        self.root.after(0, lambda: self.prog_lbl.config(text="Idle"))
    # ---------------- MISSING METHODS ----------------
    def load_editor_images(self, filter_text=None):
        """Reset and start lazy loading process."""
        for widget in self.img_list_frame.winfo_children():
            widget.destroy()
        self.editor_items = []
        
        # Reset lazy load state
        self.all_filtered_paths = []
        self.loaded_count = 0
        self.is_loading_more = False
        
        if not self.current_editor_folder: return
        
        # Start worker to find/filter paths
        threading.Thread(target=self._find_paths_worker, args=(filter_text,), daemon=True).start()

    def _find_paths_worker(self, filter_text):
        """Background thread to find and filter paths."""
        paths = []
        # Support both case patterns
        exts = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP", "*.BMP"]
        for ext in exts:
            paths.extend(self.current_editor_folder.glob(ext))
        
        # Unique paths sorted
        paths = sorted(list(set(paths)))
        
        if filter_text:
            filtered = []
            for p in paths:
                # Try image.txt
                txt = p.with_suffix(".txt")
                if not txt.exists():
                    # Try image.ext.txt
                    txt = Path(str(p) + ".txt")
                
                if txt.exists():
                    try:
                        content = txt.read_text(encoding="utf-8", errors="ignore").lower()
                        if filter_text.lower() in content:
                            filtered.append(p)
                    except: pass
            paths = filtered
            
        self.root.after(0, lambda: self._init_lazy_list(paths))

    def _init_lazy_list(self, paths):
        """Initialize list on main thread."""
        self.all_filtered_paths = paths
        self.load_more_items()

    def load_more_items(self):
        """Load next batch of items."""
        if self.is_loading_more or self.loaded_count >= len(self.all_filtered_paths):
            return
            
        self.is_loading_more = True
        batch_size = 20
        end = min(self.loaded_count + batch_size, len(self.all_filtered_paths))
        batch_paths = self.all_filtered_paths[self.loaded_count:end]
        
        # Create placeholders immediately
        img_labels = []
        for p in batch_paths:
            lbl = self.create_editor_item(p, None) # Placeholder
            img_labels.append((p, lbl))
            
        self.loaded_count = end
        
        # Start background thread to load thumbnails for this batch
        threading.Thread(target=self._load_thumbnails_batch, args=(img_labels,), daemon=True).start()

    def _load_thumbnails_batch(self, items):
        """Load thumbnails for a batch and update UI."""
        for path, lbl in items:
            # Check cache
            thumb = self.thumb_cache.get(str(path))
            if not thumb:
                try:
                    img = Image.open(path)
                    img.thumbnail((self.thumb_size, self.thumb_size))
                    thumb = img
                    self.thumb_cache[str(path)] = thumb
                except: pass
            
            if thumb:
                # Update UI on main thread
                self.root.after(0, lambda l=lbl, t=thumb: self._update_thumb(l, t))
            
            # Tiny sleep to keep UI responsive during heavy processing
            # import time; time.sleep(0.005) 
        
        self.is_loading_more = False

    def _update_thumb(self, label, thumb_img):
        try:
            photo = ImageTk.PhotoImage(thumb_img)
            label.configure(image=photo, text="")
            label.image = photo
        except: pass

    def apply_editor_filter(self):
        filter_text = self.editor_filter_var.get().strip()
        self.load_editor_images(filter_text)

    def clear_editor_filter(self):
        self.editor_filter_var.set("")
        self.load_editor_images()
    
    
    def field(self, parent, label, var, browse):
        """Create a modern input field with optional browse button."""
        if label:
            tk.Label(parent, text=label, bg=BG, fg=DIM, font=("Sans", 9, "bold")).pack(anchor="w", pady=(0, 4))
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(0, 12))
        entry = tk.Entry(row, textvariable=var, bg=INPUT, fg=TEXT, bd=0, relief="flat",
                        highlightthickness=0, font=("Sans", 10), insertbackground=CYAN)
        entry.pack(side="left", fill="x", expand=True, ipady=8)
        # Add hover effects to entry
        entry.bind("<Enter>", lambda e: entry.config(bg=HOVER))
        entry.bind("<Leave>", lambda e: entry.config(bg=INPUT))
        if browse:
            browse_btn = tk.Button(row, text="…", bg=CARD, fg=TEXT, bd=0, relief="flat",
                                  highlightthickness=0, width=4, font=("Sans", 10, "bold"),
                                  command=lambda: var.set(self._folder_picker(f"Select {label}")))
            browse_btn.pack(side="right", padx=(4, 0))
            # Add hover effects to browse button
            browse_btn.bind("<Enter>", lambda e: browse_btn.config(bg=HOVER))
            browse_btn.bind("<Leave>", lambda e: browse_btn.config(bg=CARD))

    def btn(self, parent, text, color, cmd):
        """Create a modern styled button with hover effects."""
        btn = tk.Button(parent, text=text, bg=color, fg="white", bd=0, relief="flat",
                        highlightthickness=0, font=("Sans", 9, "bold"), cursor="hand2", command=cmd,
                        activebackground=HOVER, activeforeground=TEXT)
        btn.pack(fill="x", ipady=8)
        # Add hover effects
        btn.bind("<Enter>", lambda e: btn.config(bg=HOVER))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
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
