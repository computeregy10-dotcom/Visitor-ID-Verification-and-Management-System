"""
ID Check System
================
A complete ID verification system in a single Python file.

Features:
  • Scan ID via image file or live webcam with brightness boost
  • OCR-based field extraction (Name, DOB, ID No, Expiry)
  • Age & expiry validation
  • Face Match tab:
      - ID photo side:  load from file  OR  capture via webcam
      - Live face side: capture via webcam
      - Compares both faces → match → unique access token
                              no match → "This is not the real ID holder"
  • Full audit log stored in SQLite

Install dependencies:
    pip install opencv-python pytesseract pillow numpy insightface onnxruntime

Install Tesseract OCR engine:
    Windows : https://github.com/UB-Mannheim/tesseract/wiki
    Linux   : sudo apt install tesseract-ocr
    macOS   : brew install tesseract
"""

import tkinter as tk
import insightface
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import re
import sqlite3
import datetime
import threading
import secrets
import string

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # adjust as needed

DB_PATH                   = "id_check_log.db"
MINIMUM_AGE               = 18          # BUG 1 FIX: was commented out, causing NameError
CAM_BRIGHTNESS            = 1.6
FACE_SIMILARITY_THRESHOLD = 0.55
CASCADE_FACE              = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ─────────────────────────────────────────────
# TOKEN
# ─────────────────────────────────────────────

def generate_token(length=20):
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

# BUG 3 FIX: Removed duplicate pytesseract.pytesseract.tesseract_cmd assignment
# that previously appeared here a second time.

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, name TEXT, dob TEXT, id_number TEXT,
        expiry TEXT, age INTEGER, result TEXT, reason TEXT, image_path TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS face_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, match INTEGER, similarity REAL, token TEXT, reason TEXT)""")
    conn.commit(); conn.close()

def log_check(name, dob, id_number, expiry, age, result, reason, image_path=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO checks
        (timestamp,name,dob,id_number,expiry,age,result,reason,image_path)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         name, dob, id_number, expiry, age, result, reason, image_path))
    conn.commit(); conn.close()

def log_face_check(match, similarity, token, reason):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO face_checks (timestamp,match,similarity,token,reason)
        VALUES (?,?,?,?,?)""",
        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         int(match), round(similarity,4), token, reason))
    conn.commit(); conn.close()

def fetch_log():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,name TEXT,dob TEXT,id_number TEXT,
            expiry TEXT,age INTEGER,result TEXT,reason TEXT,image_path TEXT)""")
        c.execute("SELECT timestamp,name,dob,age,result,reason FROM checks ORDER BY id DESC LIMIT 100")
        rows = c.fetchall(); conn.close(); return rows
    except sqlite3.Error: return []

def fetch_face_log():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS face_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,match INTEGER,similarity REAL,token TEXT,reason TEXT)""")
        c.execute("SELECT timestamp,match,similarity,token,reason FROM face_checks ORDER BY id DESC LIMIT 100")
        rows = c.fetchall(); conn.close(); return rows
    except sqlite3.Error: return []

# ─────────────────────────────────────────────
# IMAGE / OCR
# ─────────────────────────────────────────────

_app = None
def get_face_app():
    global _app
    if _app is None:
        _app = insightface.app.FaceAnalysis(name="buffalo_s")
        _app.prepare(ctx_id=-1)  # -1 = CPU
    return _app

def boost_brightness(pil_img, factor=CAM_BRIGHTNESS):
    return ImageEnhance.Brightness(pil_img).enhance(factor)

def preprocess_image(pil_img):
    gray  = pil_img.convert("L")
    sharp = gray.filter(ImageFilter.SHARPEN)
    enh   = ImageEnhance.Contrast(sharp).enhance(2.5)
    w, h  = enh.size
    return enh.resize((w*2, h*2), Image.LANCZOS)

def extract_text_from_image(pil_img):
    return pytesseract.image_to_string(preprocess_image(pil_img), config="--oem 3 --psm 6")

def parse_id_fields(text):
    fields = {"name":"","dob":"","id_number":"","expiry":""}
    lines  = [l.strip() for l in text.splitlines() if l.strip()]
    found  = []
    for p in [r"\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b",
              r"\b(\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b",
              r"\b(\d{2}\s+\w{3,9}\s+\d{4})\b",
              r"\b(\w{3,9}\s+\d{1,2},?\s+\d{4})\b"]:
        for m in re.finditer(p, text, re.IGNORECASE): found.append(m.group(1))
    if found:        fields["dob"]    = found[0]
    if len(found)>1: fields["expiry"] = found[1]
    for p in [r"(?:ID|No\.?|Number|Doc)[:\s#]*([A-Z0-9\-]{6,20})",
              r"\b([A-Z]{1,3}[0-9]{6,12})\b", r"\b([0-9]{8,15})\b"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: fields["id_number"] = m.group(1).strip(); break
    for p in [r"(?:Name|Full Name)[:\s]+([A-Za-z\s\-\'\.]{3,50})",
              r"(?:Surname|Last Name)[:\s]+([A-Za-z\s\-\'\.]{2,30})"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: fields["name"] = m.group(1).strip(); break
    if not fields["name"]:
        for line in lines:
            if re.match(r"^[A-Z][A-Z\s\-\'\.]{5,40}$", line):
                fields["name"] = line; break
    return fields

def parse_date(s):
    for fmt in ["%d/%m/%Y","%m/%d/%Y","%Y-%m-%d","%d-%m-%Y","%d.%m.%Y",
                "%Y/%m/%d","%d %B %Y","%d %b %Y","%B %d, %Y","%b %d, %Y",
                "%B %d %Y","%b %d %Y"]:
        try: return datetime.datetime.strptime(s.strip(), fmt).date()
        except ValueError: pass
    return None

def calculate_age(dob):
    today = datetime.date.today()
    age   = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day): age -= 1
    return age

def verify_id(fields):
    age = None
    if not fields["id_number"]: return "DENIED","ID number not detected.", age
    if not fields["dob"]:       return "DENIED","Date of birth not detected.", age
    dob = parse_date(fields["dob"])
    if dob is None: return "DENIED",f"Cannot parse DOB: {fields['dob']}", age
    age = calculate_age(dob)
    if age < MINIMUM_AGE: return "DENIED",f"Under age ({age} < {MINIMUM_AGE}).", age  # BUG 1 now works
    if fields["expiry"]:
        exp = parse_date(fields["expiry"])
        if exp and exp < datetime.date.today():
            return "DENIED",f"ID expired on {fields['expiry']}.", age
    return "APPROVED",f"Age {age} — ID valid.", age

# ─────────────────────────────────────────────
# FACE COMPARISON
# ─────────────────────────────────────────────

_face_cascade = None
def get_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(CASCADE_FACE)
    return _face_cascade

def pil_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def detect_largest_face(cv_img):
    cascade = get_cascade()
    gray    = cv2.equalizeHist(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY))
    faces   = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) == 0: return None
    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
    pad = int(min(w,h)*0.15)
    return cv_img[max(0,y-pad):min(cv_img.shape[0],y+h+pad),
                  max(0,x-pad):min(cv_img.shape[1],x+w+pad)]

def compare_faces(id_pil, live_pil):
    try:
        app  = get_face_app()
        img1 = np.array(id_pil.convert("RGB"))
        img2 = np.array(live_pil.convert("RGB"))

        faces1 = app.get(img1)
        faces2 = app.get(img2)

        if not faces1: return False, 0.0, "No face detected in ID photo."
        if not faces2: return False, 0.0, "No face detected in webcam."

        e1 = faces1[0].embedding / np.linalg.norm(faces1[0].embedding)
        e2 = faces2[0].embedding / np.linalg.norm(faces2[0].embedding)

        sim   = float(np.dot(e1, e2))
        match = sim >= 0.35

        if match:
            return True,  sim, f"Faces match (similarity {sim:.0%})."
        else:
            return False, sim, f"⚠ This is not the real ID holder (similarity {sim:.0%})."
    except Exception as e:
        return False, 0.0, f"Error: {str(e)}"

def draw_face_box(pil_img, label="", color=(0,220,80)):
    cv_img  = pil_to_cv(pil_img)
    cascade = get_cascade()
    gray    = cv2.equalizeHist(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY))
    faces   = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    for (x,y,w,h) in faces:
        cv2.rectangle(cv_img,(x,y),(x+w,y+h),color,2)
        if label:
            cv2.putText(cv_img,label,(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────

class IDCheckApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ID Verification System")

        # ── Scan tab webcam state ──
        self._cam          = None
        self._cam_running  = False
        self._live_frame   = None
        self.current_pil_image = None

        # ── Face Match — ID photo webcam state ──
        self._id_cam          = None
        self._id_cam_running  = False
        self._id_live_frame   = None
        self.id_photo_pil     = None

        # ── Face Match — live face webcam state ──
        self._fm_cam          = None
        self._fm_cam_running  = False
        self._fm_live_frame   = None
        self.live_face_pil    = None

        # Image refs (prevent GC)
        self._tk_img     = None
        self._fm_id_tk   = None
        self._fm_live_tk = None

        self._setup_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Style ──────────────────────────────────

    def _setup_style(self):
        self.configure(bg="#0f1117")
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook",     background="#0f1117", borderwidth=0)
        s.configure("TNotebook.Tab", background="#1c1f2e", foreground="#888",
                    padding=[14,6],  font=("Courier",10,"bold"))
        s.map("TNotebook.Tab",
              background=[("selected","#1a73e8")],
              foreground=[("selected","#fff")])
        s.configure("TFrame",  background="#0f1117")
        s.configure("TLabel",  background="#0f1117", foreground="#ccc",
                    font=("Courier",10))
        s.configure("TButton", background="#1a73e8", foreground="#fff",
                    font=("Courier",10,"bold"), borderwidth=0, padding=8)
        s.map("TButton", background=[("active","#0d5bbf"),("disabled","#2a2a2a")])
        s.configure("Capture.TButton", background="#e65100", foreground="#fff",
                    font=("Courier",10,"bold"), borderwidth=0, padding=8)
        s.map("Capture.TButton",
              background=[("active","#bf360c"),("disabled","#2a2a2a")])
        s.configure("Green.TButton", background="#2e7d32", foreground="#fff",
                    font=("Courier",10,"bold"), borderwidth=0, padding=8)
        s.map("Green.TButton",
              background=[("active","#1b5e20"),("disabled","#2a2a2a")])
        s.configure("Header.TLabel", background="#0f1117", foreground="#fff",
                    font=("Courier",14,"bold"))
        s.configure("Result.TLabel", background="#0f1117",
                    font=("Courier",20,"bold"))
        s.configure("Token.TLabel",  background="#1c1f2e", foreground="#ffd54f",
                    font=("Courier",13,"bold"), padding=10)
        s.configure("Treeview",      background="#1c1f2e", foreground="#ccc",
                    fieldbackground="#1c1f2e", rowheight=24, font=("Courier",9))
        s.configure("Treeview.Heading", background="#252840", foreground="#aaa",
                    font=("Courier",9,"bold"))

    # ── Top layout ─────────────────────────────

    def _build_ui(self):
        hdr = ttk.Frame(self); hdr.pack(fill="x", pady=(16,4), padx=20)
        ttk.Label(hdr, text="🪪  ID VERIFICATION SYSTEM",
                  style="Header.TLabel").pack(side="left")
        tk.Frame(self, height=1, bg="#1a73e8").pack(fill="x", padx=20)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=12)

        self._build_scanner_tab()
        self._build_face_tab()
        self._build_manual_tab()
        self._build_log_tab()

    # ─────────────────────────────────────────
    # SCANNER TAB
    # ─────────────────────────────────────────

    def _build_scanner_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  📷  Scan ID  ")

        left  = ttk.Frame(tab); left.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        right = ttk.Frame(tab); right.pack(side="right", fill="y", padx=(0,12), pady=12)

        self.canvas = tk.Canvas(left, width=480, height=300, bg="#1c1f2e",
                                highlightthickness=1, highlightbackground="#333")
        self.canvas.pack()
        self._placeholder(self.canvas, 480, 300, "📷  Press 'Live Webcam' to start camera")

        self.cam_status_var = tk.StringVar(value="Camera off")
        ttk.Label(left, textvariable=self.cam_status_var,
                  foreground="#555", font=("Courier",9)).pack(anchor="w", pady=(3,0))

        bf = ttk.Frame(left); bf.pack(pady=6, fill="x")
        ttk.Button(bf, text="📂  Load Image",    command=self._load_image).pack(side="left",padx=(0,6))
        ttk.Button(bf, text="📷  Live Webcam",   command=self._start_webcam).pack(side="left",padx=(0,6))
        self.capture_btn = ttk.Button(bf, text="📸  Capture", style="Capture.TButton",
                                      command=self._capture_frame, state="disabled")
        self.capture_btn.pack(side="left",padx=(0,6))
        ttk.Button(bf, text="⏹  Stop",          command=self._stop_webcam).pack(side="left",padx=(0,6))
        ttk.Button(bf, text="🔍  Scan & Verify", command=self._scan_and_verify).pack(side="left")

        ttk.Label(left, text="OCR Output:").pack(anchor="w", pady=(8,2))
        self.ocr_text = tk.Text(left, width=55, height=7, bg="#1c1f2e", fg="#88c0d0",
                                insertbackground="#fff", font=("Courier",9),
                                borderwidth=0, relief="flat", wrap="word")
        self.ocr_text.pack(fill="x")

        ttk.Label(right, text="Parsed Fields", foreground="#888").pack(anchor="w")
        tk.Frame(right, height=1, bg="#333").pack(fill="x", pady=4)
        self.field_vars = {}
        for label in ("Name","Date of Birth","ID Number","Expiry"):
            key = label.lower().replace(" ","_").replace("date_of_birth","dob")
            row = ttk.Frame(right); row.pack(fill="x", pady=3)
            ttk.Label(row, text=f"{label}:", width=14, foreground="#888").pack(side="left")
            var = tk.StringVar(value="—"); self.field_vars[key] = var
            ttk.Label(row, textvariable=var, foreground="#fff", width=22).pack(side="left")

        self.age_var = tk.StringVar(value="—")
        row = ttk.Frame(right); row.pack(fill="x", pady=3)
        ttk.Label(row, text="Age:", width=14, foreground="#888").pack(side="left")
        ttk.Label(row, textvariable=self.age_var, foreground="#fff", width=22).pack(side="left")

        tk.Frame(right, height=1, bg="#333").pack(fill="x", pady=10)
        self.result_var   = tk.StringVar(value="AWAITING")
        self.result_label = ttk.Label(right, textvariable=self.result_var, style="Result.TLabel")
        self.result_label.pack()
        self.reason_var   = tk.StringVar(value="Load or capture an ID to begin.")
        ttk.Label(right, textvariable=self.reason_var, wraplength=220,
                  foreground="#aaa", font=("Courier",9)).pack(pady=4)

    # ─────────────────────────────────────────
    # FACE MATCH TAB
    # ─────────────────────────────────────────

    def _build_face_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  🔍  Face Match  ")

        ttk.Label(tab,
                  text="Step 1 — Get ID photo (load file or capture via webcam)  |  Step 2 — Capture your live face",
                  foreground="#888", font=("Courier",9)).pack(anchor="w", padx=16, pady=(12,4))

        top = ttk.Frame(tab); top.pack(fill="x", padx=16, pady=4)

        # ── ID PHOTO PANEL ──────────────────────
        id_frame = ttk.Frame(top); id_frame.pack(side="left", padx=(0,20))

        ttk.Label(id_frame, text="ID Photo", foreground="#aaa",
                  font=("Courier",10,"bold")).pack()

        self.fm_id_canvas = tk.Canvas(id_frame, width=280, height=200, bg="#1c1f2e",
                                      highlightthickness=1, highlightbackground="#1a73e8")
        self.fm_id_canvas.pack()
        self._placeholder(self.fm_id_canvas, 280, 200, "📂 load  or  📷 webcam")

        self.id_cam_status = tk.StringVar(value="")
        ttk.Label(id_frame, textvariable=self.id_cam_status,
                  foreground="#555", font=("Courier",8)).pack(anchor="w", pady=(2,0))

        id_row1 = ttk.Frame(id_frame); id_row1.pack(pady=(4,2))
        ttk.Button(id_row1, text="📂  Load File",
                   command=self._id_load_file).pack(side="left", padx=(0,6))
        ttk.Button(id_row1, text="📷  Webcam",
                   command=self._id_start_webcam).pack(side="left")

        id_row2 = ttk.Frame(id_frame); id_row2.pack(pady=(0,4))
        self.id_capture_btn = ttk.Button(id_row2, text="📸  Capture ID",
                                         style="Capture.TButton",
                                         command=self._id_capture_frame,
                                         state="disabled")
        self.id_capture_btn.pack(side="left", padx=(0,6))
        ttk.Button(id_row2, text="⏹  Stop",
                   command=self._id_stop_webcam).pack(side="left")

        # ── LIVE FACE PANEL ──────────────────────
        live_frame = ttk.Frame(top); live_frame.pack(side="left")

        ttk.Label(live_frame, text="Your Live Face", foreground="#aaa",
                  font=("Courier",10,"bold")).pack()

        self.fm_live_canvas = tk.Canvas(live_frame, width=280, height=200, bg="#1c1f2e",
                                        highlightthickness=1, highlightbackground="#333")
        self.fm_live_canvas.pack()
        self._placeholder(self.fm_live_canvas, 280, 200, "📷 Start webcam →")

        self.fm_cam_status = tk.StringVar(value="Camera off")
        ttk.Label(live_frame, textvariable=self.fm_cam_status,
                  foreground="#555", font=("Courier",8)).pack(anchor="w", pady=(2,0))

        live_btns = ttk.Frame(live_frame); live_btns.pack(pady=6)
        ttk.Button(live_btns, text="📷  Start Webcam",
                   command=self._fm_start_webcam).pack(side="left", padx=(0,6))
        self.fm_capture_btn = ttk.Button(live_btns, text="📸  Capture Face",
                                         style="Capture.TButton",
                                         command=self._fm_capture_frame,
                                         state="disabled")
        self.fm_capture_btn.pack(side="left", padx=(0,6))
        ttk.Button(live_btns, text="⏹  Stop",
                   command=self._fm_stop_webcam).pack(side="left")

        # ── Compare ──────────────────────────────
        sep = tk.Frame(tab, height=1, bg="#333"); sep.pack(fill="x", padx=16, pady=(10,6))
        ttk.Button(tab, text="⚖️   Compare Faces & Get Token",
                   style="Green.TButton",
                   command=self._fm_compare).pack(padx=16, anchor="w")

        # ── Result ───────────────────────────────
        res = tk.Frame(tab, bg="#1c1f2e"); res.pack(fill="x", padx=16, pady=8)

        self.fm_result_var = tk.StringVar(value="")
        self.fm_result_lbl = ttk.Label(res, textvariable=self.fm_result_var,
                                       style="Result.TLabel")
        self.fm_result_lbl.pack(pady=(10,2))

        self.fm_reason_var = tk.StringVar(value="Load the ID photo, capture your face, then press Compare.")
        ttk.Label(res, textvariable=self.fm_reason_var, wraplength=600,
                  foreground="#aaa", font=("Courier",10)).pack(pady=(0,6))

        self.fm_token_var = tk.StringVar(value="")
        self.fm_token_lbl = ttk.Label(res, textvariable=self.fm_token_var,
                                      style="Token.TLabel")
        self.fm_token_lbl.pack(pady=4)

        ttk.Button(res, text="📋  Copy Token", command=self._fm_copy_token).pack(pady=(0,10))

    # ─────────────────────────────────────────
    # MANUAL ENTRY TAB
    # ─────────────────────────────────────────

    def _build_manual_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  ✏️  Manual Entry  ")
        frame = ttk.Frame(tab); frame.pack(expand=True, pady=40, padx=60)
        ttk.Label(frame, text="Manual ID Verification",
                  style="Header.TLabel").grid(row=0,column=0,columnspan=2,pady=(0,20),sticky="w")
        self.manual_fields = {}
        for i,(label,key) in enumerate([
            ("Full Name","name"),("Date of Birth (DD/MM/YYYY)","dob"),
            ("ID Number","id_number"),("Expiry Date (DD/MM/YYYY)","expiry")],start=1):
            ttk.Label(frame,text=label+":").grid(row=i,column=0,sticky="e",pady=6,padx=(0,12))
            var = tk.StringVar()
            tk.Entry(frame,textvariable=var,width=28,bg="#1c1f2e",fg="#fff",
                     insertbackground="#fff",font=("Courier",11),relief="flat",bd=4
                     ).grid(row=i,column=1,sticky="w",pady=6)
            self.manual_fields[key] = var
        ttk.Button(frame,text="✅  Verify",
                   command=self._manual_verify).grid(row=6,column=0,columnspan=2,pady=20)
        self.manual_result_var = tk.StringVar(value="")
        self.manual_result_lbl = ttk.Label(frame,textvariable=self.manual_result_var,
                                           style="Result.TLabel")
        self.manual_result_lbl.grid(row=7,column=0,columnspan=2)
        self.manual_reason_var = tk.StringVar(value="")
        ttk.Label(frame,textvariable=self.manual_reason_var,
                  foreground="#aaa",font=("Courier",9)).grid(row=8,column=0,columnspan=2)

    # ─────────────────────────────────────────
    # LOG TAB
    # ─────────────────────────────────────────

    def _build_log_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  📋  Log  ")
        ctrl = ttk.Frame(tab); ctrl.pack(fill="x",padx=12,pady=(12,4))
        ttk.Button(ctrl,text="🔄  Refresh",command=self._refresh_log).pack(side="left")
        ttk.Button(ctrl,text="🗑️  Clear All",command=self._clear_log).pack(side="left",padx=8)

        ttk.Label(tab,text="ID Scan Log",foreground="#888",
                  font=("Courier",9,"bold")).pack(anchor="w",padx=12)
        cols=("Timestamp","Name","DOB","Age","Result","Reason"); widths=[140,160,100,50,80,260]
        self.tree = ttk.Treeview(tab,columns=cols,show="headings",height=8)
        for col,w in zip(cols,widths):
            self.tree.heading(col,text=col)
            self.tree.column(col,width=w,anchor="center" if w<120 else "w")
        self.tree.tag_configure("approved",foreground="#4caf50")
        self.tree.tag_configure("denied",  foreground="#f44336")
        sc1=ttk.Scrollbar(tab,orient="vertical",command=self.tree.yview)
        self.tree.configure(yscrollcommand=sc1.set)
        self.tree.pack(side="left",fill="x",expand=True,padx=(12,0),pady=(0,6))
        sc1.pack(side="left",fill="y",pady=(0,6))

        ttk.Label(tab,text="Face Match Log",foreground="#888",
                  font=("Courier",9,"bold")).pack(anchor="w",padx=12)
        fcols=("Timestamp","Match","Similarity","Token","Reason"); fw=[140,80,90,210,230]
        self.ftree = ttk.Treeview(tab,columns=fcols,show="headings",height=8)
        for col,w in zip(fcols,fw):
            self.ftree.heading(col,text=col)
            self.ftree.column(col,width=w,anchor="center" if w<120 else "w")
        self.ftree.tag_configure("match",  foreground="#4caf50")
        self.ftree.tag_configure("nomatch",foreground="#f44336")
        sc2=ttk.Scrollbar(tab,orient="vertical",command=self.ftree.yview)
        self.ftree.configure(yscrollcommand=sc2.set)
        self.ftree.pack(side="left",fill="x",expand=True,padx=(12,0),pady=(0,12))
        sc2.pack(side="left",fill="y",pady=(0,12))
        self._refresh_log()

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _placeholder(self, canvas, w, h, text):
        canvas.delete("all")
        canvas.create_text(w//2, h//2, text=text, fill="#444", font=("Courier",11))

    def _show_on_canvas(self, canvas, pil_img, w, h, label="", box_color=(0,220,80)):
        """Annotate face boxes, display image on canvas, and return the PhotoImage reference."""
        annotated = draw_face_box(pil_img, label, box_color) if label else pil_img.copy()
        annotated.thumbnail((w, h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(annotated)
        canvas.delete("all")
        canvas.create_image(w//2, h//2, anchor="center", image=tk_img)
        # BUG 4 FIX: overlay text is now drawn AFTER image on same canvas call, no second
        # delete("all") is needed; the image reference is returned and stored by the caller.
        return tk_img

    # ─────────────────────────────────────────
    # SCANNER TAB — webcam
    # ─────────────────────────────────────────

    def _start_webcam(self):
        # BUG 6 FIX: stop any other active streams before opening camera
        self._stop_all_other_cams("scan")
        if self._cam_running: return
        self._cam = cv2.VideoCapture(0)
        if not self._cam.isOpened():
            messagebox.showerror("Webcam","Could not open webcam."); self._cam=None; return
        self._cam_running = True
        self.capture_btn.configure(state="normal")
        self.cam_status_var.set("🟢  Live — press 📸 Capture to freeze")
        threading.Thread(target=self._feed_loop, daemon=True).start()

    def _feed_loop(self):
        while self._cam_running:
            ret,frame = self._cam.read()
            if not ret: break
            pil = boost_brightness(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            self._live_frame = pil
            disp = pil.copy(); disp.thumbnail((480,300),Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(disp)
            self.after(0, self._update_scan_canvas, tk_img)
        if self._cam: self._cam.release(); self._cam=None

    def _update_scan_canvas(self, tk_img):
        self._tk_img = tk_img   # keep reference alive
        self.canvas.delete("all")
        self.canvas.create_image(240,150,anchor="center",image=self._tk_img)

    def _capture_frame(self):
        if self._live_frame is None: return
        self.current_pil_image = self._live_frame.copy()
        self._cam_running = False
        self.capture_btn.configure(state="disabled")
        self.cam_status_var.set("📸  Captured — press 🔍 Scan & Verify")
        frozen = self.current_pil_image.copy(); frozen.thumbnail((480,300),Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(frozen)
        self.canvas.delete("all")
        self.canvas.create_image(240,150,anchor="center",image=self._tk_img)
        self.canvas.create_rectangle(0,270,480,300,fill="#000",stipple="gray50")
        self.canvas.create_text(240,285,
                                text="✅  Captured — press 🔍 Scan & Verify",
                                fill="#4caf50",font=("Courier",9,"bold"))

    def _stop_webcam(self):
        self._cam_running = False
        self.capture_btn.configure(state="disabled")
        self.cam_status_var.set("Camera off")
        self._live_frame = None
        self.after(80, lambda: self._placeholder(
            self.canvas, 480, 300, "📷  Press 'Live Webcam' to start camera"))

    def _load_image(self):
        self._stop_webcam()
        path = filedialog.askopenfilename(
            filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files","*.*")])
        if not path: return
        try:
            img = Image.open(path); self.current_pil_image = img
            disp = img.copy(); disp.thumbnail((480,300),Image.LANCZOS)
            self._tk_img = ImageTk.PhotoImage(disp)
            self.canvas.delete("all")
            self.canvas.create_image(240,150,anchor="center",image=self._tk_img)
        except Exception as e: messagebox.showerror("Error",str(e))

    def _scan_and_verify(self):
        if self.current_pil_image is None:
            messagebox.showwarning("No image","Load or capture an ID image first."); return
        self.result_var.set("SCANNING…"); self.reason_var.set("Running OCR…")
        self.update_idletasks()
        def run():
            try:
                text   = extract_text_from_image(self.current_pil_image)
                fields = parse_id_fields(text)
                result,reason,age = verify_id(fields)
                self.after(0,lambda: self._display_scan_result(text,fields,result,reason,age))
                log_check(fields["name"],fields["dob"],fields["id_number"],
                          fields["expiry"],age,result,reason)
            except pytesseract.TesseractNotFoundError:
                self.after(0,lambda: messagebox.showerror("Tesseract Missing",
                    "Tesseract OCR not found.\nInstall:\n"
                    "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "  Linux:   sudo apt install tesseract-ocr\n"
                    "  macOS:   brew install tesseract"))
                self.after(0,lambda: self.result_var.set("ERROR"))
            except Exception as e:
                self.after(0,lambda: messagebox.showerror("Error",str(e)))
                self.after(0,lambda: self.result_var.set("ERROR"))
        threading.Thread(target=run,daemon=True).start()

    def _display_scan_result(self,raw,fields,result,reason,age):
        self.ocr_text.delete("1.0","end"); self.ocr_text.insert("end",raw)
        self.field_vars["name"].set(fields["name"] or "—")
        self.field_vars["dob"].set(fields["dob"] or "—")
        self.field_vars["id_number"].set(fields["id_number"] or "—")
        self.field_vars["expiry"].set(fields["expiry"] or "—")
        self.age_var.set(str(age) if age is not None else "—")
        self.result_var.set(result); self.reason_var.set(reason)
        self.result_label.configure(foreground="#4caf50" if result=="APPROVED" else "#f44336")
        self._refresh_log()

    # ─────────────────────────────────────────
    # BUG 6 FIX: Helper that stops any active stream that isn't `caller`
    # ─────────────────────────────────────────

    def _stop_all_other_cams(self, caller):
        """Gracefully stop webcam streams that are NOT the caller, so only one
        stream uses cv2.VideoCapture(0) at a time."""
        if caller != "scan" and self._cam_running:
            self._cam_running = False
            self.capture_btn.configure(state="disabled")
            self.cam_status_var.set("Camera off (released for another stream)")
        if caller != "id" and self._id_cam_running:
            self._id_cam_running = False
            self.id_capture_btn.configure(state="disabled")
            self.id_cam_status.set("Camera off (released for another stream)")
        if caller != "live" and self._fm_cam_running:
            self._fm_cam_running = False
            self.fm_capture_btn.configure(state="disabled")
            self.fm_cam_status.set("Camera off (released for another stream)")
        # Brief sleep lets feed threads notice the flag change and release the device
        import time; time.sleep(0.15)

    # ─────────────────────────────────────────
    # FACE MATCH — ID PHOTO WEBCAM
    # ─────────────────────────────────────────

    def _id_load_file(self):
        self._id_stop_webcam()
        path = filedialog.askopenfilename(
            filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files","*.*")])
        if not path: return
        try:
            img = Image.open(path)
            self.id_photo_pil = img
            self._fm_id_tk = self._show_on_canvas(
                self.fm_id_canvas, img, 280, 200, "ID", (30,144,255))
            self.id_cam_status.set("📂  Loaded from file")
        except Exception as e: messagebox.showerror("Error",str(e))

    def _id_start_webcam(self):
        # BUG 6 FIX: stop competing streams first
        self._stop_all_other_cams("id")
        if self._id_cam_running: return
        self._id_cam = cv2.VideoCapture(0)
        if not self._id_cam.isOpened():
            messagebox.showerror("Webcam","Could not open webcam."); self._id_cam=None; return
        self._id_cam_running = True
        self.id_capture_btn.configure(state="normal")
        self.id_cam_status.set("🟢  Live — press 📸 Capture ID")
        threading.Thread(target=self._id_feed_loop, daemon=True).start()

    def _id_feed_loop(self):
        while self._id_cam_running:
            ret,frame = self._id_cam.read()
            if not ret: break
            pil = boost_brightness(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            self._id_live_frame = pil
            disp = pil.copy(); disp.thumbnail((280,200),Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(disp)
            self.after(0, self._id_update_canvas, tk_img)
        if self._id_cam: self._id_cam.release(); self._id_cam=None

    def _id_update_canvas(self, tk_img):
        self._fm_id_tk = tk_img   # keep reference alive
        self.fm_id_canvas.delete("all")
        self.fm_id_canvas.create_image(140,100,anchor="center",image=self._fm_id_tk)

    def _id_capture_frame(self):
        if self._id_live_frame is None: return
        self.id_photo_pil = self._id_live_frame.copy()
        self._id_cam_running = False
        self.id_capture_btn.configure(state="disabled")
        self.id_cam_status.set("📸  ID photo captured")
        # BUG 4 FIX: store return value then overlay text in a separate canvas call;
        # no extra delete("all") that would erase the image.
        self._fm_id_tk = self._show_on_canvas(
            self.fm_id_canvas, self.id_photo_pil, 280, 200, "ID", (30,144,255))
        self.fm_id_canvas.create_text(140,190,
                                      text="✅  ID captured",
                                      fill="#4caf50",font=("Courier",8,"bold"))

    def _id_stop_webcam(self):
        self._id_cam_running = False
        self.id_capture_btn.configure(state="disabled")
        self._id_live_frame = None
        self.id_cam_status.set("")
        self.after(80, lambda: self._placeholder(
            self.fm_id_canvas, 280, 200, "📂 load  or  📷 webcam"))

    # ─────────────────────────────────────────
    # FACE MATCH — LIVE FACE WEBCAM
    # ─────────────────────────────────────────

    def _fm_start_webcam(self):
        # BUG 6 FIX: stop competing streams first
        self._stop_all_other_cams("live")
        if self._fm_cam_running: return
        self._fm_cam = cv2.VideoCapture(0)
        if not self._fm_cam.isOpened():
            messagebox.showerror("Webcam","Could not open webcam."); self._fm_cam=None; return
        self._fm_cam_running = True
        self.fm_capture_btn.configure(state="normal")
        self.fm_cam_status.set("🟢  Live — press 📸 Capture Face")
        threading.Thread(target=self._fm_feed_loop, daemon=True).start()

    def _fm_feed_loop(self):
        while self._fm_cam_running:
            ret,frame = self._fm_cam.read()
            if not ret: break
            pil = boost_brightness(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            self._fm_live_frame = pil
            disp = pil.copy(); disp.thumbnail((280,200),Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(disp)
            self.after(0, self._fm_update_live_canvas, tk_img)
        if self._fm_cam: self._fm_cam.release(); self._fm_cam=None

    def _fm_update_live_canvas(self, tk_img):
        self._fm_live_tk = tk_img   # keep reference alive
        self.fm_live_canvas.delete("all")
        self.fm_live_canvas.create_image(140,100,anchor="center",image=self._fm_live_tk)

    def _fm_capture_frame(self):
        if self._fm_live_frame is None: return
        self.live_face_pil = self._fm_live_frame.copy()
        self._fm_cam_running = False
        self.fm_capture_btn.configure(state="disabled")
        self.fm_cam_status.set("📸  Face captured. Camera stopped.")
        # BUG 4 FIX: store return value then overlay text without erasing image
        self._fm_live_tk = self._show_on_canvas(
            self.fm_live_canvas, self.live_face_pil, 280, 200, "YOU", (0,220,80))
        self.fm_live_canvas.create_text(140,190,
                                        text="✅  Face captured",
                                        fill="#4caf50",font=("Courier",8,"bold"))

    def _fm_stop_webcam(self):
        self._fm_cam_running = False
        self.fm_capture_btn.configure(state="disabled")
        self.fm_cam_status.set("Camera off")
        self._fm_live_frame = None
        self.after(80, lambda: self._placeholder(self.fm_live_canvas,280,200,"📷 Start webcam →"))

    # ─────────────────────────────────────────
    # FACE COMPARISON + TOKEN
    # ─────────────────────────────────────────

    def _fm_compare(self):
        if self.id_photo_pil is None:
            messagebox.showwarning("Missing","Please load or capture the ID photo first."); return
        if self.live_face_pil is None:
            messagebox.showwarning("Missing","Please capture your live face first."); return
        self.fm_result_var.set("COMPARING…")
        self.fm_reason_var.set("Analysing faces…")
        self.fm_token_var.set("")
        self.update_idletasks()
        def run():
            match, sim, reason = compare_faces(self.id_photo_pil, self.live_face_pil)
            if match:
                token        = generate_token()
                result_text  = "✅  MATCH"
                color        = "#4caf50"
                token_display= f"ACCESS TOKEN:  {token}"
            else:
                token        = ""
                result_text  = "❌  NO MATCH"
                color        = "#f44336"
                token_display= ""
            log_face_check(match, sim, token, reason)
            self.after(0,lambda: self._display_face_result(result_text,color,reason,token_display))
        threading.Thread(target=run,daemon=True).start()

    def _display_face_result(self,result_text,color,reason,token_display):
        self.fm_result_var.set(result_text)
        self.fm_result_lbl.configure(foreground=color)
        self.fm_reason_var.set(reason)
        self.fm_token_var.set(token_display)
        self._refresh_log()

    def _fm_copy_token(self):
        token = self.fm_token_var.get()
        if not token or "TOKEN:" not in token:
            messagebox.showinfo("No token","No token to copy yet."); return
        self.clipboard_clear(); self.clipboard_append(token.split("TOKEN:")[-1].strip())
        messagebox.showinfo("Copied","Token copied to clipboard!")

    # ─────────────────────────────────────────
    # MANUAL VERIFY
    # ─────────────────────────────────────────

    def _manual_verify(self):
        fields = {k: v.get().strip() for k,v in self.manual_fields.items()}
        result,reason,age = verify_id(fields)
        self.manual_result_var.set(result); self.manual_reason_var.set(reason)
        self.manual_result_lbl.configure(
            foreground="#4caf50" if result=="APPROVED" else "#f44336")
        log_check(fields["name"],fields["dob"],fields["id_number"],
                  fields["expiry"],age,result,reason)
        self._refresh_log()

    # ─────────────────────────────────────────
    # LOG
    # ─────────────────────────────────────────

    def _refresh_log(self):
        for r in self.tree.get_children(): self.tree.delete(r)
        for row in fetch_log():
            ts,name,dob,age,result,reason = row
            tag = "approved" if result=="APPROVED" else "denied"
            self.tree.insert("","end",
                values=(ts,name or "—",dob or "—",age or "—",result,reason),tags=(tag,))

        for r in self.ftree.get_children(): self.ftree.delete(r)
        # BUG 2 FIX: face log insert was commented out — now restored
        for row in fetch_face_log():
            ts,match,sim,token,reason = row
            tag = "match" if match else "nomatch"
            self.ftree.insert("","end",
                values=(ts, "✅ MATCH" if match else "❌ NO MATCH",
                        f"{sim:.0%}", token or "—", reason), tags=(tag,))

    def _clear_log(self):
        if messagebox.askyesno("Clear Logs","Delete ALL log entries?"):
            conn=sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM checks"); conn.execute("DELETE FROM face_checks")
            conn.commit(); conn.close(); self._refresh_log()

    # ─────────────────────────────────────────
    # CLOSE
    # ─────────────────────────────────────────

    def _on_close(self):
        self._cam_running = self._id_cam_running = self._fm_cam_running = False
        self.after(120, self.destroy)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    app = IDCheckApp()
    app.geometry("1280x720")
    app.mainloop()
