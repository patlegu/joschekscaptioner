#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joschek’s Captioner v29 - Stable Edition (Hotfix 2)
Fixes: 
- Critical crash in Editor scrolling logic (Tkinter TclError).
- Attribute Error on startup (missing editor_filtered init).
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import subprocess
import os
import shutil
import threading
import signal
import glob
import base64
import json
import time
import requests
import re
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from openai import OpenAI
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ---------------- CONSTANTS ----------------
CONFIG_FILE = Path.home() / ".config" / "joschek_captioner.json"

BACKEND_NATIVE = "Native (llama-server)"
BACKEND_OLLAMA = "Ollama"
BACKEND_LMSTUDIO = "LM Studio"

# Extended COCO Classes for YOLO
YOLO_CLASSES = {
    "Person": 0, "Bicycle": 1, "Car": 2, "Motorcycle": 3, "Airplane": 4, 
    "Bus": 5, "Train": 6, "Truck": 7, "Boat": 8, "Bird": 14, 
    "Cat": 15, "Dog": 16, "Horse": 17, "Sheep": 18, "Cow": 19, 
    "Backpack": 24, "Umbrella": 25, "Handbag": 26, "Tie": 27, 
    "Suitcase": 28, "Chair": 56, "Couch": 57, "Potted Plant": 58, 
    "Bed": 59, "Dining Table": 60, "Toilet": 61, "TV": 62, 
    "Laptop": 63, "Mouse": 64, "Remote": 65, "Keyboard": 66, 
    "Cell Phone": 67, "Microwave": 68, "Oven": 69, "Toaster": 70, 
    "Sink": 71, "Refrigerator": 72, "Book": 73, "Clock": 74, "Vase": 75
}

DEFAULTS = {
    "backend": BACKEND_NATIVE,
    "port": "8080",
    "server_binary": "./llama-server",
    "model_file": "",
    "projector_file": "",
    "selected_model_name": "",
    "context": "8192",
    "gpu_layers": "99",
    "crop_class": "Person",
    "crop_method": "pad",
    "pad_color": "#000000",
    "last_prompt": "Describe this image in detail for an AI training dataset. Focus on clothing, background, textures, and lighting."
}

# ---------------- THEME ----------------
BG = "#1a1b26"
CARD = "#24283b"
INPUT = "#2a2d3e"
TEXT = "#f8f8f2"
DIM = "#6272a4"
BLUE = "#5294e2"
GREEN = "#50fa7b"
RED = "#ff5555"

# ---------------- CORE CONFIG ----------------
class Config:
    def __init__(self) -> None:
        self.config_dir: Path = CONFIG_FILE.parent
        self.data: Dict[str, Any] = self.load()
    
    def load(self) -> Dict[str, Any]:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if not CONFIG_FILE.exists(): 
            self.create_template()
            return DEFAULTS.copy()
        try:
            raw = CONFIG_FILE.read_text(encoding="utf-8")
            clean_lines = []
            for line in raw.splitlines():
                if not (line.strip().startswith("#") or line.strip().startswith("//")):
                    clean_lines.append(line)
            clean = "\n".join(clean_lines)
            clean = re.sub(r',(\s*})', r'\1', clean)
            return json.loads(clean)
        except Exception as e:
            print(f"Config Error: {e}")
            return DEFAULTS.copy()

    def save(self) -> None: pass
    def get(self, key: str, default: Any = None) -> Any: return self.data.get(key, default if default is not None else DEFAULTS.get(key))
    def set(self, key: str, value: Any) -> None: self.data[key] = value

    def create_template(self) -> None:
        txt = """{
    # JOSCHEK'S CAPTIONER CONFIG
    # "backend": "Native (llama-server)",
    # "port": "8080"
}"""
        try: CONFIG_FILE.write_text(txt, encoding="utf-8")
        except: pass

    def export_clean_json(self, path: Path) -> None:
        try: path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except: pass

# ---------------- WIDGETS ----------------
class QueueItem(tk.Frame):
    def __init__(self, parent: tk.Widget, path: Path, remove_cb: Callable, config: Config) -> None:
        super().__init__(parent, bg=CARD)
        self.folder_path = path
        main = tk.Frame(self, bg=CARD); main.pack(fill="both", expand=True, padx=10, pady=5)
        h = tk.Frame(main, bg=CARD); h.pack(fill="x")
        tk.Label(h, text=path.name, bg=CARD, fg=TEXT, font=("Sans", 9, "bold")).pack(side="left")
        self.lbl_status = tk.Label(h, text="Ready", bg=CARD, fg=DIM, font=("Sans", 8)); self.lbl_status.pack(side="left", padx=10)
        close = tk.Label(h, text="×", bg=CARD, fg=DIM, cursor="hand2", font=("Sans", 12)); close.pack(side="right")
        close.bind("<Button-1>", lambda e: remove_cb(self))
        self.prompt = tk.Text(main, height=2, bg=INPUT, fg=TEXT, bd=0, font=("Sans", 8), wrap="word")
        self.prompt.insert("1.0", str(config.get("last_prompt"))); self.prompt.pack(fill="x", pady=(5,0))
        self.prompt.bind("<KeyRelease>", lambda e: config.set("last_prompt", self.prompt.get("1.0", "end-1c").strip()))
    def set_status(self, s: str, m: str) -> None: self.lbl_status.config(text=m, fg={"processing":BLUE,"done":GREEN,"error":RED}.get(s, DIM))
    def get_prompt(self) -> str: return self.prompt.get("1.0", "end-1c").strip()

class ScrollFrame(tk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=BG)
        c = tk.Canvas(self, bg=BG, highlightthickness=0); s = tk.Scrollbar(self, command=c.yview)
        self.content = tk.Frame(c, bg=BG); self.win = c.create_window((0, 0), window=self.content, anchor="nw")
        c.configure(yscrollcommand=s.set); c.pack(side="left", fill="both", expand=True); s.pack(side="right", fill="y")
        c.bind("<Configure>", lambda e: c.itemconfig(self.win, width=e.width))
        self.content.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
        c.bind_all("<MouseWheel>", lambda e: c.yview_scroll(-1 if e.delta>0 else 1, "units"))

# ---------------- MAIN APP ----------------
class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Joschek's Captioner v29 [Stable]")
        root.geometry("1100x850")
        root.configure(bg=BG)
        root.option_add("*Font", ("Sans", 10))

        self.config = Config()
        self.style_setup()
        
        self.server_proc = None
        self.server_ready = False
        
        self.build_ui()
        self.update_ui_for_backend(str(self.config.get("backend")))
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def style_setup(self) -> None:
        s = ttk.Style(); s.theme_use("clam")
        s.configure("TProgressbar", background=BLUE, troughcolor=BG, borderwidth=0)
        s.configure("TCombobox", fieldbackground=INPUT, background=INPUT, foreground=TEXT, arrowcolor=TEXT)
        self.root.option_add("*TCombobox*Listbox*Background", CARD)
        self.root.option_add("*TCombobox*Listbox*Foreground", TEXT)

    def build_ui(self) -> None:
        main = tk.Frame(self.root, bg=BG); main.pack(fill="both", expand=True)
        bar = tk.Frame(main, bg=BG); bar.pack(side="top", fill="x", padx=20, pady=10)
        self.content = tk.Frame(main, bg=BG); self.content.pack(side="bottom", fill="both", expand=True)
        
        self.tabs = {}; self.btns = {}; self.curr_tab = None
        for n in ["Server", "Batch Captioning", "Manual Edit", "Filter & Move", "Auto Cropping"]:
            self.tabs[n] = tk.Frame(self.content, bg=BG)
            b = tk.Label(bar, text=n, bg=BG, fg=DIM, font=("Sans", 10, "bold"), cursor="hand2", padx=12, pady=5)
            b.pack(side="left"); b.bind("<Button-1>", lambda e, x=n: self.switch_tab(x))
            self.btns[n] = b
        tk.Frame(main, bg=INPUT, height=1).pack(side="top", fill="x")
        self.switch_tab("Server")

        self.build_server_tab()
        self.build_batch_tab()
        self.build_editor_tab()
        self.build_filter_tab()
        self.build_crop_tab()

    def switch_tab(self, name: str) -> None:
        if self.curr_tab: self.tabs[self.curr_tab].pack_forget(); self.btns[self.curr_tab].config(fg=DIM)
        self.curr_tab = name; self.tabs[name].pack(fill="both", expand=True); self.btns[name].config(fg=BLUE)

    # --- TAB 1: SERVER ---
    def build_server_tab(self) -> None:
        f = tk.Frame(self.tabs["Server"], bg=BG); f.pack(fill="both", expand=True, padx=30, pady=20)
        
        tk.Label(f, text="Backend Engine", bg=BG, fg=DIM, font=("Sans", 9, "bold")).pack(anchor="w")
        self.backend_var = tk.StringVar(value=str(self.config.get("backend")))
        be = tk.Frame(f, bg=BG); be.pack(fill="x", pady=(0, 15))
        cb = ttk.Combobox(be, textvariable=self.backend_var, values=[BACKEND_NATIVE, BACKEND_OLLAMA, BACKEND_LMSTUDIO], state="readonly")
        cb.pack(side="left", fill="x", expand=True, ipady=4); cb.bind("<<ComboboxSelected>>", lambda e: self.on_backend_change())

        self.settings_frame = tk.Frame(f, bg=BG); self.settings_frame.pack(fill="x")
        
        # Native
        self.native_frame = tk.Frame(self.settings_frame, bg=BG)
        self.sv_bin = tk.StringVar(value=str(self.config.get("server_binary")))
        self.sv_mod = tk.StringVar(value=str(self.config.get("model_file")))
        self.sv_prj = tk.StringVar(value=str(self.config.get("projector_file")))
        self.sv_port = tk.StringVar(value=str(self.config.get("port")))
        self.sv_ctx = tk.StringVar(value=str(self.config.get("context")))
        self.sv_gpu = tk.StringVar(value=str(self.config.get("gpu_layers")))
        
        self.field(self.native_frame, "Server Binary", self.sv_bin, True, "file")
        self.field(self.native_frame, "Model (.gguf)", self.sv_mod, True, "file")
        self.field(self.native_frame, "Projector", self.sv_prj, True, "file")
        
        rn = tk.Frame(self.native_frame, bg=BG); rn.pack(fill="x")
        for l, v in [("Context", self.sv_ctx), ("GPU Layers", self.sv_gpu)]:
            c = tk.Frame(rn, bg=BG); c.pack(side="left", fill="x", expand=True, padx=5)
            tk.Label(c, text=l, bg=BG, fg=DIM).pack(anchor="w")
            tk.Entry(c, textvariable=v, bg=INPUT, fg=TEXT, bd=0).pack(fill="x", ipady=5)
        
        # Remote
        self.remote_frame = tk.Frame(self.settings_frame, bg=BG)
        self.sv_model_name = tk.StringVar(value=str(self.config.get("selected_model_name")))
        rr = tk.Frame(self.remote_frame, bg=BG); rr.pack(fill="x")
        tk.Label(rr, text="Select Model", bg=BG, fg=DIM).pack(anchor="w")
        self.cb_models = ttk.Combobox(rr, textvariable=self.sv_model_name, state="normal"); self.cb_models.pack(fill="x", pady=(0,10))
        self.btn(self.remote_frame, "Refresh Models", CARD, self.fetch_remote_models).pack(anchor="e")

        # Port & Controls
        cp = tk.Frame(f, bg=BG); cp.pack(fill="x", pady=10)
        tk.Label(cp, text="API Port", bg=BG, fg=DIM).pack(anchor="w")
        tk.Entry(cp, textvariable=self.sv_port, bg=INPUT, fg=TEXT, bd=0).pack(fill="x", ipady=5)

        ctrl = tk.Frame(f, bg=BG); ctrl.pack(fill="x", pady=15)
        self.btn_action = self.btn(ctrl, "Start Server", BLUE, self.toggle_server)
        self.btn_action.pack(side="left", fill="x", expand=True)
        self.lbl_health = tk.Label(ctrl, text="OFFLINE", bg=BG, fg=DIM, font=("Sans", 8, "bold")); self.lbl_health.pack(side="left", padx=10)
        self.btn(ctrl, "Export Config", CARD, self.export_profile).pack(side="right")

        self.log = tk.Text(f, height=8, bg="#111", fg=GREEN, font=("Monospace", 8), bd=0); self.log.pack(fill="both", expand=True)
        
        for v, k in [(self.sv_bin, "server_binary"), (self.sv_mod, "model_file"), (self.sv_port, "port"), (self.backend_var, "backend")]:
            v.trace_add("write", lambda *_, val=v, key=k: self.config.set(key, val.get()))

    def on_backend_change(self) -> None: self.update_ui_for_backend(self.backend_var.get())

    def update_ui_for_backend(self, mode: str) -> None:
        self.native_frame.pack_forget(); self.remote_frame.pack_forget()
        if mode == BACKEND_NATIVE:
            self.native_frame.pack(fill="x"); self.btn_action.config(text="Start Server", bg=BLUE)
            if not self.sv_port.get(): self.sv_port.set("8080")
        else:
            self.remote_frame.pack(fill="x"); self.btn_action.config(text="Connect", bg=GREEN)
            if not self.sv_port.get(): self.sv_port.set("11434" if mode==BACKEND_OLLAMA else "1234")
            self.root.after(100, self.fetch_remote_models)

    def fetch_remote_models(self) -> None: threading.Thread(target=self._fetch_models, daemon=True).start()
    def _fetch_models(self) -> None:
        try:
            client = OpenAI(base_url=f"http://localhost:{self.sv_port.get()}/v1", api_key="x")
            names = [m.id for m in client.models.list().data]
            self.root.after(0, lambda: self.cb_models.config(values=names))
            if names and not self.sv_model_name.get(): self.root.after(0, lambda: self.cb_models.current(0))
        except: pass

    def toggle_server(self) -> None:
        if self.backend_var.get() == BACKEND_NATIVE:
            if self.server_proc: self.stop_native()
            else: self.start_native()
        else: self.check_health()

    def start_native(self) -> None:
        cmd = [self.sv_bin.get(), "-m", self.sv_mod.get(), "--port", self.sv_port.get(), "-ngl", self.sv_gpu.get(), "-b", "512"]
        if self.sv_prj.get(): cmd.extend(["--mmproj", self.sv_prj.get()])
        kwargs = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'preexec_fn': os.setsid}
        try:
            self.server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **kwargs)
            self.btn_action.config(text="Stop", bg=RED)
            threading.Thread(target=self.monitor_log, daemon=True).start()
            threading.Thread(target=self.check_health, daemon=True).start()
        except Exception as e: self.log_add(str(e))

    def stop_native(self) -> None:
        if self.server_proc:
            try: self.server_proc.terminate()
            except: pass
            self.server_proc = None
        self.lbl_health.config(text="OFFLINE", fg=DIM); self.btn_action.config(text="Start Server", bg=BLUE)

    def monitor_log(self) -> None:
        if self.server_proc and self.server_proc.stdout:
            for line in iter(self.server_proc.stdout.readline, ""):
                if line: self.root.after(0, lambda l=line.strip(): self.log_add(l))

    def check_health(self) -> None:
        url = f"http://localhost:{self.sv_port.get()}/v1/models"
        for _ in range(30):
            try:
                if requests.get(url, timeout=1).status_code == 200:
                    self.server_ready = True; self.root.after(0, lambda: self.lbl_health.config(text="ONLINE", fg=GREEN))
                    return
            except: pass
            time.sleep(1)
        self.server_ready = False; self.root.after(0, lambda: self.lbl_health.config(text="UNREACHABLE", fg=RED))

    def export_profile(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p: self.config.export_clean_json(Path(p))
    def log_add(self, msg: str) -> None: self.log.insert("end", msg+"\n"); self.log.see("end")

    # --- TAB 2: BATCH ---
    def build_batch_tab(self) -> None:
        f = tk.Frame(self.tabs["Batch Captioning"], bg=BG); f.pack(fill="both", expand=True, padx=25, pady=15)
        top = tk.Frame(f, bg=BG); top.pack(fill="x", pady=10)
        self.btn(top, "+ Folder", BLUE, self.batch_add).pack(side="left")
        self.btn_run = self.btn(top, "Start", BLUE, self.batch_toggle); self.btn_run.pack(side="left", padx=10)
        self.batch_ow = tk.BooleanVar()
        tk.Checkbutton(top, text="Overwrite", variable=self.batch_ow, bg=BG, fg=TEXT, selectcolor=INPUT).pack(side="right")
        self.queue_ui = ScrollFrame(f); self.queue_ui.pack(fill="both", expand=True)
        self.batch_bar = ttk.Progressbar(f, mode="determinate"); self.batch_bar.pack(fill="x", pady=10)
        self.queue = []; self.batch_running = False

    def batch_add(self) -> None:
        p = filedialog.askdirectory()
        if p: self.queue.append(QueueItem(self.queue_ui.content, Path(p), lambda i: (i.destroy(), self.queue.remove(i)), self.config)); self.queue[-1].pack(fill="x", pady=2)
    def batch_toggle(self) -> None:
        if self.batch_running: self.batch_running = False; self.btn_run.config(text="Start", bg=BLUE)
        else:
            if not self.server_ready: return messagebox.showerror("Err", "Backend offline")
            self.batch_running = True; self.btn_run.config(text="Stop", bg=RED); threading.Thread(target=self.batch_loop, daemon=True).start()
    def batch_loop(self) -> None:
        try: client = OpenAI(base_url=f"http://localhost:{self.sv_port.get()}/v1", api_key="x")
        except: return
        model = Path(self.sv_mod.get()).stem if self.backend_var.get() == BACKEND_NATIVE else self.sv_model_name.get()
        for item in self.queue:
            if not self.batch_running: break
            item.set_status("processing", "Working...")
            imgs = glob.glob(os.path.join(str(item.folder_path), "*.jpg")) + glob.glob(os.path.join(str(item.folder_path), "*.png"))
            done = 0
            for img in imgs:
                if not self.batch_running: break
                try:
                    with open(img, "rb") as f: b64 = base64.b64encode(f.read()).decode()
                    res = client.chat.completions.create(model=model, messages=[{"role":"user","content":[{"type":"text","text":item.get_prompt()},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}], max_tokens=300)
                    Path(img).with_suffix(".txt").write_text(res.choices[0].message.content, encoding="utf-8")
                    done += 1
                except: pass
                self.root.after(0, lambda d=done, t=len(imgs), i=item: [self.batch_bar.configure(value=(d/t)*100), i.set_status("processing", f"{d}/{t}")])
            item.set_status("done", "Done")
        self.batch_running = False; self.root.after(0, lambda: self.btn_run.config(text="Start", bg=BLUE))

    # --- TAB 3: EDITOR (FIXED) ---
    def build_editor_tab(self) -> None:
        f = tk.Frame(self.tabs["Manual Edit"], bg=BG); f.pack(fill="both", expand=True, padx=25, pady=15)
        top = tk.Frame(f, bg=BG); top.pack(fill="x", pady=10)
        self.btn(top, "Open Folder", BLUE, self.editor_load_folder).pack(side="left")
        tk.Label(top, text="Filter:", bg=BG, fg=DIM).pack(side="right", padx=5)
        self.ed_filter_var = tk.StringVar(); e = tk.Entry(top, textvariable=self.ed_filter_var, bg=INPUT, fg=TEXT, bd=0); e.pack(side="right", fill="y", ipady=5); e.bind("<Return>", lambda e: self.editor_apply_filter())
        self.lbl_ed_info = tk.Label(top, text="No folder", bg=BG, fg=DIM); self.lbl_ed_info.pack(side="left", padx=15)
        
        self.ed_canvas = tk.Canvas(f, bg=BG, highlightthickness=0)
        # --- FIX: Store reference to scrollbar ---
        self.ed_scrollbar = tk.Scrollbar(f, command=self.ed_canvas.yview)
        
        self.ed_frame = tk.Frame(self.ed_canvas, bg=BG)
        self.ed_win = self.ed_canvas.create_window((0,0), window=self.ed_frame, anchor="nw")
        
        self.ed_canvas.configure(yscrollcommand=self.editor_on_scroll)
        self.ed_canvas.pack(side="left", fill="both", expand=True)
        self.ed_scrollbar.pack(side="right", fill="y")
        
        self.ed_frame.bind("<Configure>", lambda e: self.ed_canvas.configure(scrollregion=self.ed_canvas.bbox("all")))
        self.ed_canvas.bind("<Configure>", lambda e: self.ed_canvas.itemconfig(self.ed_win, width=e.width))
        self.ed_canvas.bind_all("<MouseWheel>", lambda e: self.ed_canvas.yview_scroll(-1 if e.delta>0 else 1, "units"))
        
        # --- FIX: INITIALIZE VARIABLES ---
        self.thumb_cache = {}
        self.editor_paths = []
        self.editor_filtered = [] # ADDED MISSING INIT
        self.editor_loaded_idx = 0
        self.editor_loading = False

    def editor_load_folder(self) -> None:
        p = filedialog.askdirectory()
        if not p: return
        self.editor_folder = Path(p); self.lbl_ed_info.config(text="Scanning...")
        threading.Thread(target=self.editor_scan, daemon=True).start()
    def editor_scan(self) -> None:
        files = []; 
        for ext in ["*.jpg","*.png"]: files.extend(list(self.editor_folder.glob(ext)))
        self.editor_paths = sorted(files); self.editor_filtered = self.editor_paths
        self.root.after(0, self.editor_reset)
    def editor_apply_filter(self) -> None:
        kw = self.ed_filter_var.get().lower().strip()
        if not kw: self.editor_filtered = self.editor_paths
        else:
            res = []
            for p in self.editor_paths:
                t = p.with_suffix(".txt")
                if t.exists() and kw in t.read_text(encoding="utf-8").lower(): res.append(p)
            self.editor_filtered = res
        self.editor_reset()
    def editor_reset(self) -> None:
        for w in self.ed_frame.winfo_children(): w.destroy()
        self.editor_loaded_idx = 0; self.lbl_ed_info.config(text=f"{len(self.editor_filtered)} images")
        self.editor_load_more()
    def editor_on_scroll(self, *args) -> None:
        # --- FIX: Update scrollbar, NOT canvas ---
        self.ed_scrollbar.set(*args)
        # --- FIX: Check if filtered list exists and has content before loading more ---
        if self.editor_filtered and len(args) > 1 and float(args[1]) > 0.9 and not self.editor_loading:
            self.editor_load_more()
    def editor_load_more(self) -> None:
        if self.editor_loaded_idx >= len(self.editor_filtered): return
        self.editor_loading = True
        end = min(self.editor_loaded_idx + 20, len(self.editor_filtered))
        for p in self.editor_filtered[self.editor_loaded_idx:end]: self.editor_create_item(p)
        self.editor_loaded_idx = end; self.editor_loading = False
    def editor_create_item(self, path: Path) -> None:
        row = tk.Frame(self.ed_frame, bg=CARD, pady=5); row.pack(fill="x", pady=5, padx=5)
        holder = tk.Frame(row, bg=BG, width=150, height=150); holder.pack_propagate(False); holder.pack(side="left", padx=5)
        lbl_img = tk.Label(holder, text="...", bg=BG, fg=DIM); lbl_img.pack(expand=True)
        lbl_img.bind("<Double-Button-1>", lambda e, p=path: self.editor_zoom(p))
        txt = path.with_suffix(".txt"); c = txt.read_text(encoding="utf-8", errors="ignore") if txt.exists() else ""
        t = tk.Text(row, height=6, bg=INPUT, fg=TEXT, bd=0, wrap="word"); t.insert("1.0", c); t.pack(side="left", fill="both", expand=True, padx=10)
        t.bind("<KeyRelease>", lambda e: txt.write_text(t.get("1.0", "end-1c"), encoding="utf-8"))
        threading.Thread(target=self.editor_thumb, args=(path, lbl_img), daemon=True).start()
    def editor_thumb(self, path: Path, lbl: tk.Label) -> None:
        sp = str(path)
        if sp not in self.thumb_cache:
            try: i = Image.open(path); i.thumbnail((140, 140)); self.thumb_cache[sp] = ImageTk.PhotoImage(i)
            except: return
        self.root.after(0, lambda: lbl.config(image=self.thumb_cache[sp], text=""))
    def editor_zoom(self, path: Path) -> None:
        tl = tk.Toplevel(self.root); tl.title(path.name); tl.configure(bg=BG); tl.geometry("800x800")
        try: i = Image.open(path); i.thumbnail((800, 800)); ph = ImageTk.PhotoImage(i); l = tk.Label(tl, image=ph, bg=BG); l.image=ph; l.pack(expand=True)
        except: pass

    # --- TAB 4: FILTER ---
    def build_filter_tab(self) -> None:
        f = tk.Frame(self.tabs["Filter & Move"], bg=BG); f.pack(fill="both", expand=True, padx=30, pady=20)
        self.fil_src = tk.StringVar(); self.fil_tgt = tk.StringVar(); self.fil_kw = tk.StringVar()
        self.field(f, "Source Folder", self.fil_src, True, "folder")
        self.field(f, "Target Folder", self.fil_tgt, True, "folder")
        tk.Label(f, text="Keyword", bg=BG, fg=DIM).pack(anchor="w"); tk.Entry(f, textvariable=self.fil_kw, bg=INPUT, fg=TEXT, bd=0).pack(fill="x", pady=(0, 15), ipady=6)
        self.btn(f, "Move Matches", BLUE, self.filter_run)
        self.fil_log = tk.Label(f, text="Idle", bg=BG, fg=DIM); self.fil_log.pack(pady=10)
    def filter_run(self) -> None:
        if not self.fil_src.get() or not self.fil_tgt.get() or not self.fil_kw.get(): return
        self.fil_log.config(text="Searching..."); threading.Thread(target=self.filter_worker, daemon=True).start()
    def filter_worker(self) -> None:
        src = Path(self.fil_src.get()); tgt = Path(self.fil_tgt.get()); kw = self.fil_kw.get().lower(); c = 0
        for t in src.glob("*.txt"):
            try:
                if kw in t.read_text(encoding="utf-8").lower():
                    shutil.move(str(t), str(tgt/t.name))
                    for e in [".png", ".jpg"]:
                        if (i:=t.with_suffix(e)).exists(): shutil.move(str(i), str(tgt/i.name))
                    c += 1
            except: pass
        self.root.after(0, lambda: self.fil_log.config(text=f"Moved {c} pairs"))

    # --- TAB 5: AUTO CROP ---
    def build_crop_tab(self) -> None:
        f = tk.Frame(self.tabs["Auto Cropping"], bg=BG); f.pack(fill="both", expand=True, padx=30, pady=20)
        self.crop_in = tk.StringVar(); self.crop_out = tk.StringVar()
        self.field(f, "Input Folder", self.crop_in, True, "folder")
        self.field(f, "Output Folder", self.crop_out, True, "folder")
        
        cfg = tk.Frame(f, bg=BG); cfg.pack(fill="x", pady=15)
        # Class
        c1 = tk.Frame(cfg, bg=BG); c1.pack(side="left", fill="x", expand=True, padx=5)
        tk.Label(c1, text="Object to Crop", bg=BG, fg=DIM, font=("Sans", 8)).pack(anchor="w")
        self.crop_cls = tk.StringVar(value=self.config.get("crop_class"))
        cls_cb = ttk.Combobox(c1, textvariable=self.crop_cls, values=sorted(list(YOLO_CLASSES.keys())), state="readonly")
        cls_cb.pack(fill="x", ipady=5); cls_cb.bind("<<ComboboxSelected>>", lambda e: self.config.set("crop_class", self.crop_cls.get()))
        # Method
        c2 = tk.Frame(cfg, bg=BG); c2.pack(side="left", fill="x", expand=True, padx=5)
        tk.Label(c2, text="Resizing Method", bg=BG, fg=DIM, font=("Sans", 8)).pack(anchor="w")
        self.crop_method = tk.StringVar(value=self.config.get("crop_method"))
        meth_cb = ttk.Combobox(c2, textvariable=self.crop_method, values=["crop", "pad"], state="readonly")
        meth_cb.pack(fill="x", ipady=5); meth_cb.bind("<<ComboboxSelected>>", lambda e: self.config.set("crop_method", self.crop_method.get()))
        # Color
        c3 = tk.Frame(cfg, bg=BG); c3.pack(side="left", fill="x", expand=True, padx=5)
        tk.Label(c3, text="Pad Color", bg=BG, fg=DIM, font=("Sans", 8)).pack(anchor="w")
        self.pad_col = tk.StringVar(value=self.config.get("pad_color"))
        cr = tk.Frame(c3, bg=BG); cr.pack(fill="x")
        self.btn_col = tk.Button(cr, bg=self.pad_col.get(), width=3, command=self.pick_color, bd=0); self.btn_col.pack(side="left", padx=(0,5))
        tk.Entry(cr, textvariable=self.pad_col, bg=INPUT, fg=TEXT, bd=0).pack(side="left", fill="x")

        self.btn(f, "Start Advanced Crop", BLUE, self.crop_run).pack(pady=20)
        self.crop_log = tk.Label(f, text="Idle", bg=BG, fg=DIM); self.crop_log.pack()

    def pick_color(self) -> None:
        c = colorchooser.askcolor(color=self.pad_col.get(), title="Padding Color")
        if c[1]: self.pad_col.set(c[1]); self.btn_col.config(bg=c[1]); self.config.set("pad_color", c[1])

    def crop_run(self) -> None:
        if not YOLO: return messagebox.showerror("Err", "Install ultralytics")
        threading.Thread(target=self.crop_worker, daemon=True).start()

    def crop_worker(self) -> None:
        model = YOLO("yolov8n-seg.pt")
        src = Path(self.crop_in.get()); dst = Path(self.crop_out.get()); dst.mkdir(exist_ok=True)
        tid = YOLO_CLASSES.get(self.crop_cls.get(), 0)
        method = self.crop_method.get()
        hc = self.pad_col.get().lstrip('#')
        rgb = tuple(int(hc[i:i+2], 16) for i in (0, 2, 4)); bgr = (rgb[2], rgb[1], rgb[0])
        imgs = list(src.glob("*.jpg")) + list(src.glob("*.png"))
        for i, path in enumerate(imgs):
            try:
                im = cv2.imread(str(path)); res = model.predict(im, conf=0.5, classes=[tid], verbose=False)
                if not res[0].masks: continue
                for idx, mask in enumerate(res[0].masks):
                    x,y,w,h = cv2.boundingRect(mask.xy[0].astype(np.int32))
                    m = 20; x=max(0,x-m); y=max(0,y-m); w=min(im.shape[1]-x, w+m*2); h=min(im.shape[0]-y, h+m*2)
                    crop = im[y:y+h, x:x+w]; target = 1024
                    if method == "pad":
                        h_c, w_c = crop.shape[:2]; scale = target/max(h_c, w_c)
                        nw, nh = int(w_c*scale), int(h_c*scale); resized = cv2.resize(crop, (nw, nh))
                        dw, dh = target-nw, target-nh; final = cv2.copyMakeBorder(resized, dh//2, dh-(dh//2), dw//2, dw-(dw//2), cv2.BORDER_CONSTANT, value=bgr)
                    else: final = cv2.resize(crop, (target, target))
                    cv2.imwrite(str(dst / f"{path.stem}_{idx}.jpg"), final)
                self.root.after(0, lambda x=i: self.crop_log.config(text=f"Processed {x+1}/{len(imgs)}"))
            except: pass
        self.root.after(0, lambda: self.crop_log.config(text="Done."))

    # --- UTILS ---
    def field(self, parent: tk.Widget, label: str, var: tk.StringVar, browse: bool, mode: str) -> None:
        tk.Label(parent, text=label, bg=BG, fg=DIM, font=("Sans", 8, "bold")).pack(anchor="w")
        r = tk.Frame(parent, bg=BG); r.pack(fill="x", pady=(0, 10))
        tk.Entry(r, textvariable=var, bg=INPUT, fg=TEXT, bd=0).pack(side="left", fill="x", expand=True, ipady=6)
        if browse:
            c = lambda: var.set(filedialog.askopenfilename() if mode == "file" else filedialog.askdirectory())
            tk.Button(r, text="...", command=c, bg=CARD, fg=TEXT, bd=0).pack(side="right", padx=5)
    def btn(self, parent: tk.Widget, text: str, color: str, cmd: Callable[[], None]) -> tk.Button:
        b = tk.Button(parent, text=text, bg=color, fg="white", bd=0, font=("Sans", 9, "bold"), command=cmd)
        b.pack(pady=5, ipady=5); return b
    def on_close(self) -> None:
        if self.server_proc: self.server_proc.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()