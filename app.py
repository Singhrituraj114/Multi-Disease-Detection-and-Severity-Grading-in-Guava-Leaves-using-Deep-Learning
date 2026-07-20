import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
from inference import run_yolo, run_gradcam, disease_info

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_REGULAR = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_BOLD    = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuavaGuard AI — Leaf Pathology Intelligence",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# PDF REPORT ENGINE
# ──────────────────────────────────────────────────────────────────
class DiagnosticReport(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu",  "",  FONT_REGULAR, uni=True)
        self.add_font("DejaVu", "B",  FONT_BOLD,    uni=True)

    def header(self):
        self.set_fill_color(10, 30, 15)
        self.rect(0, 0, 210, 45, "F")
        # accent bar
        self.set_fill_color(0, 200, 100)
        self.rect(0, 43, 210, 2, "F")
        self.set_y(10)
        self.set_text_color(0, 220, 110)
        self.set_font("DejaVu", "B", 20)
        self.cell(0, 10, "DIAGNOSTIC REPORT", ln=True, align="C")
        self.set_font("DejaVu", "", 9)
        self.set_text_color(160, 220, 180)
        self.cell(0, 7, "Multi-Disease Detection & Severity Analytics", ln=True, align="C")
        self.ln(14)

    def footer(self):
        self.set_y(-16)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(130, 160, 140)
        self.cell(0, 10,
            f"GuavaGuard AI  ·  Page {self.page_no()}  ·  Generated {datetime.now().strftime('%d %b %Y, %H:%M')}",
            0, 0, "C")

    def section_title(self, text, r=0, g=150, b=80):
        self.set_font("DejaVu", "B", 12)
        self.set_text_color(r, g, b)
        self.set_fill_color(10, 30, 15)
        self.cell(0, 9, text, ln=True, fill=True)
        self.set_draw_color(0, 180, 90)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def kv(self, key, val):
        self.set_font("DejaVu", "B", 10)
        self.set_text_color(60, 160, 100)
        self.set_x(12)
        self.cell(38, 6, key, ln=0)
        self.set_font("DejaVu", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, str(val))


def generate_detailed_pdf(diseases, sev, level, bgr_img):
    pdf = DiagnosticReport()
    pdf.add_page()

    # ── SECTION I: Visual Scan ──────────────────────────────────
    pdf.section_title("I.  VISUAL DIAGNOSTIC SCAN")
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    try:
        img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(tmp)
        os.close(fd)
        start_y = pdf.get_y()
        pdf.image(tmp, x=10, y=start_y + 2, w=115)
        img_h = 115 * bgr_img.shape[0] / bgr_img.shape[1]
        pdf.set_y(start_y + img_h + 8)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    # ── SECTION II: Telemetry ───────────────────────────────────
    pdf.section_title("II.  SYSTEM TELEMETRY")
    pdf.kv("Pathogen Load:", f"{sev:.2f}%")
    pdf.kv("Severity Level:", level)
    pdf.kv("Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pdf.kv("Diseases Found:", str(len(diseases)))
    pdf.ln(6)

    # ── SECTION III: Disease Details ───────────────────────────
    pdf.section_title("III.  DETECTED PATHOLOGIES")
    for i, d in enumerate(diseases, 1):
        info = disease_info.get(d, {})
        pdf.set_fill_color(230, 255, 240)
        pdf.set_font("DejaVu", "B", 11)
        pdf.set_text_color(0, 100, 50)
        pdf.cell(0, 8, f"  {i}. {d.replace('_', ' ').title()}", ln=True, fill=True)
        pdf.ln(1)
        for key, field in [
            ("Description:", "description"),
            ("Cause:",       "cause"),
            ("Impact:",      "impact"),
            ("Treatment:",   "treatment"),
            ("Organic:",     "organic"),
            ("Prevention:",  "prevention"),
            ("Future-ready:","future_safety"),
        ]:
            pdf.kv(key, info.get(field, "N/A"))
        pdf.ln(5)

    return bytes(pdf.output(dest="S"))


# ──────────────────────────────────────────────────────────────────
# GLOBAL CSS  — immersive forest / bioluminescent theme
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── CSS Variables ────────────────────────────────────────── */
:root {
  --g0: #020d06;
  --g1: #041a0c;
  --g2: #062b14;
  --g3: #0a3d1e;
  --accent: #00e87a;
  --accent2: #39ff9e;
  --accent3: #00ffc8;
  --gold: #c8ff80;
  --text: #d4f0e0;
  --muted: #6b9e80;
  --glass: rgba(0,232,122,0.06);
  --glass-border: rgba(0,232,122,0.18);
  --radius: 20px;
  --shadow: 0 24px 64px rgba(0,0,0,0.7);
}

/* ── Root Reset ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background: var(--g0) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  min-height: 100vh;
}

/* ── Animated mesh background ─────────────────────────────── */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 20% 30%, rgba(0,100,40,0.22) 0%, transparent 60%),
    radial-gradient(ellipse 60% 70% at 80% 70%, rgba(0,60,25,0.18) 0%, transparent 60%),
    radial-gradient(ellipse 40% 40% at 50% 10%, rgba(0,232,122,0.07) 0%, transparent 50%);
  pointer-events: none;
  z-index: 0;
  animation: meshPulse 12s ease-in-out infinite alternate;
}
@keyframes meshPulse {
  0%   { opacity: 0.7; }
  100% { opacity: 1.0; }
}

/* ── Floating leaf particles (CSS only) ───────────────────── */
.leaf-field {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}
.leaf {
  position: absolute;
  width: 10px;
  height: 14px;
  background: radial-gradient(ellipse at 40% 40%, #00e87a55, #00401a22);
  border-radius: 50% 0 50% 0;
  animation: leafFall linear infinite;
  opacity: 0;
}
.leaf:nth-child(1)  { left:5%;   animation-duration:18s; animation-delay:0s;   width:8px;  height:12px; }
.leaf:nth-child(2)  { left:15%;  animation-duration:22s; animation-delay:3s;   width:12px; height:16px; }
.leaf:nth-child(3)  { left:28%;  animation-duration:16s; animation-delay:1s;   width:7px;  height:10px; }
.leaf:nth-child(4)  { left:42%;  animation-duration:24s; animation-delay:5s;   width:10px; height:14px; }
.leaf:nth-child(5)  { left:55%;  animation-duration:19s; animation-delay:2s;   width:9px;  height:13px; }
.leaf:nth-child(6)  { left:68%;  animation-duration:21s; animation-delay:7s;   width:11px; height:15px; }
.leaf:nth-child(7)  { left:78%;  animation-duration:17s; animation-delay:4s;   width:8px;  height:11px; }
.leaf:nth-child(8)  { left:88%;  animation-duration:23s; animation-delay:6s;   width:13px; height:17px; }
.leaf:nth-child(9)  { left:35%;  animation-duration:20s; animation-delay:9s;   width:7px;  height:10px; }
.leaf:nth-child(10) { left:62%;  animation-duration:25s; animation-delay:11s;  width:10px; height:14px; }
@keyframes leafFall {
  0%   { transform: translateY(-40px) rotate(0deg)   translateX(0px);   opacity: 0;   }
  10%  { opacity: 0.6; }
  50%  { transform: translateY(45vh)  rotate(180deg) translateX(30px);  opacity: 0.4; }
  90%  { opacity: 0.3; }
  100% { transform: translateY(105vh) rotate(360deg) translateX(-20px); opacity: 0;   }
}

/* ── Main content layer ───────────────────────────────────── */
.block-container {
  position: relative;
  z-index: 1;
  padding: 2rem 3rem !important;
  max-width: 1400px !important;
}

/* ── Hero Header ──────────────────────────────────────────── */
.hero-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 0.4rem 1rem 1.4rem;
  position: relative;
}
.hero-title {
  font-family: 'Playfair Display', serif !important;
  font-size: clamp(2.2rem, 4vw, 3.8rem) !important;
  font-weight: 900 !important;
  line-height: 1.12 !important;
  background: linear-gradient(135deg, #00e87a 0%, #39ff9e 40%, #00ffc8 70%, #c8ff80 100%);
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  margin: 0 0 0.8rem !important;
  animation: titleShimmer 6s ease-in-out infinite;
  background-size: 200%;
}
@keyframes titleShimmer {
  0%, 100% { background-position: 0% 50%;   }
  50%       { background-position: 100% 50%; }
}
.hero-sub {
  display: block;
  width: 100%;
  text-align: center;
  font-size: 1.05rem;
  color: var(--muted);
  letter-spacing: 0.03em;
  max-width: 680px;
  margin: 0 auto 0.2rem;
  font-weight: 300;
}
.hero-divider {
  width: 120px;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  margin: 1.8rem auto 0;
  animation: dividerGlow 3s ease-in-out infinite;
}
@keyframes dividerGlow {
  0%, 100% { opacity: 0.5; width: 80px; }
  50%       { opacity: 1.0; width: 160px; }
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #030f07 0%, #041406 100%) !important;
  border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stFileUploader label {
  color: var(--accent) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
  background: rgba(0,232,122,0.04) !important;
  border: 1.5px dashed var(--glass-border) !important;
  border-radius: 14px !important;
  transition: all 0.3s ease;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
  background: rgba(0,232,122,0.09) !important;
  border-color: var(--accent) !important;
  box-shadow: 0 0 20px rgba(0,232,122,0.15);
}

/* ── Sidebar brand ────────────────────────────────────────── */
.sidebar-brand {
  text-align: center;
  padding: 1.8rem 0 1.2rem;
  border-bottom: 1px solid var(--glass-border);
  margin-bottom: 1.4rem;
}
.sidebar-brand .sb-icon {
  font-size: 2.8rem;
  display: block;
  filter: drop-shadow(0 0 14px #00e87a88);
  animation: iconSpin 8s linear infinite;
}
@keyframes iconSpin {
  0%   { filter: drop-shadow(0 0 12px #00e87a88); }
  50%  { filter: drop-shadow(0 0 22px #00ffc888); }
  100% { filter: drop-shadow(0 0 12px #00e87a88); }
}
.sidebar-brand .sb-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.4rem;
  font-weight: 700;
  background: linear-gradient(90deg, var(--accent), var(--accent3));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-top: 0.4rem;
}
.sidebar-brand .sb-version {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-top: 0.3rem;
}

/* ── Sidebar stats strip ──────────────────────────────────── */
.stat-strip {
  display: flex;
  gap: 8px;
  margin: 1rem 0;
}
.stat-pill {
  flex: 1;
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 10px 6px;
  text-align: center;
}
.stat-pill .sp-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.1rem;
  color: var(--accent);
  font-weight: 600;
  display: block;
}
.stat-pill .sp-lbl {
  font-size: 0.6rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* ── Mode radio pills ─────────────────────────────────────── */
[data-testid="stRadio"] > div {
  gap: 8px;
  flex-direction: column;
}
[data-testid="stRadio"] label {
  background: var(--glass) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  padding: 10px 16px !important;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex !important;
  align-items: center;
}
[data-testid="stRadio"] label:hover {
  background: rgba(0,232,122,0.12) !important;
  border-color: var(--accent) !important;
  transform: translateX(4px);
  box-shadow: -4px 0 16px rgba(0,232,122,0.18);
}

/* ── Columns ──────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] { gap: 1.5rem !important; }

/* ── Glass card base ──────────────────────────────────────── */
.glass-card {
  background: linear-gradient(145deg, rgba(10,40,20,0.85), rgba(5,20,10,0.92));
  border: 1px solid var(--glass-border);
  border-radius: var(--radius);
  padding: 2rem;
  position: relative;
  overflow: hidden;
  transition: transform 0.45s cubic-bezier(0.23,1,0.32,1),
              box-shadow 0.45s ease,
              border-color 0.4s ease;
  animation: cardFloat 6s ease-in-out infinite;
}
.glass-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse at 20% 0%, rgba(0,232,122,0.07), transparent 60%);
  pointer-events: none;
}
.glass-card:hover {
  transform: translateY(-6px) scale(1.01);
  box-shadow: 0 30px 80px rgba(0,0,0,0.6), 0 0 40px rgba(0,232,122,0.12);
  border-color: rgba(0,232,122,0.4);
}
@keyframes cardFloat {
  0%, 100% { transform: translateY(0);   }
  50%       { transform: translateY(-5px); }
}

/* ── Section headings ─────────────────────────────────────── */
.sec-heading {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.sec-heading::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--glass-border), transparent);
}

/* ── Scan viewer ──────────────────────────────────────────── */
.scan-shell {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--glass-border);
  background: #000;
  box-shadow: inset 0 0 60px rgba(0,0,0,0.6), 0 0 40px rgba(0,232,122,0.08);
}
.scan-shell::after {
  content: 'LIVE ANALYSIS';
  position: absolute;
  top: 12px;
  left: 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.15em;
  color: var(--accent);
  background: rgba(0,0,0,0.6);
  padding: 3px 8px;
  border-radius: 6px;
  border: 1px solid rgba(0,232,122,0.3);
  z-index: 10;
}
.scan-line-anim {
  position: absolute;
  left: 0;
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  box-shadow: 0 0 18px var(--accent), 0 0 40px rgba(0,232,122,0.3);
  animation: scanDown 3s ease-in-out infinite;
  z-index: 5;
}
@keyframes scanDown {
  0%   { top: 0%;   opacity: 0; }
  15%  { opacity: 1; }
  85%  { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}
/* corner brackets */
.scan-shell .corner { position:absolute; width:18px; height:18px; z-index:6; }
.scan-shell .corner.tl { top:8px;    left:8px;  border-top:2px solid var(--accent); border-left:2px solid var(--accent);  border-radius:4px 0 0 0; }
.scan-shell .corner.tr { top:8px;    right:8px; border-top:2px solid var(--accent); border-right:2px solid var(--accent); border-radius:0 4px 0 0; }
.scan-shell .corner.bl { bottom:8px; left:8px;  border-bottom:2px solid var(--accent); border-left:2px solid var(--accent);  border-radius:0 0 0 4px; }
.scan-shell .corner.br { bottom:8px; right:8px; border-bottom:2px solid var(--accent); border-right:2px solid var(--accent); border-radius:0 0 4px 0; }

/* ── Metric cards ─────────────────────────────────────────── */
.metric-row {
  display: flex;
  gap: 14px;
  margin-bottom: 1.4rem;
  flex-wrap: wrap;
}
.m-card {
  flex: 1;
  min-width: 120px;
  background: linear-gradient(135deg, rgba(0,232,122,0.06), rgba(0,100,50,0.1));
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 1rem 1.2rem;
  position: relative;
  overflow: hidden;
  transition: all 0.35s ease;
}
.m-card::before {
  content: '';
  position: absolute;
  top: -30px; right: -30px;
  width: 80px; height: 80px;
  background: radial-gradient(circle, rgba(0,232,122,0.12), transparent 70%);
  pointer-events: none;
}
.m-card:hover { transform: translateY(-3px); border-color: var(--accent); box-shadow: 0 12px 32px rgba(0,0,0,0.4); }
.m-card .mc-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}
.m-card .mc-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.6rem;
  font-weight: 600;
  color: var(--accent);
  line-height: 1;
}
.m-card .mc-delta {
  font-size: 0.75rem;
  color: var(--muted);
  margin-top: 4px;
}

/* ── Severity badge ───────────────────────────────────────── */
.sev-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 20px;
  border-radius: 100px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  animation: badgePulse 2.5s ease-in-out infinite;
}
.sev-healthy  { background:rgba(0,200,80,0.15);  border:1px solid rgba(0,200,80,0.4);  color:#00c850; }
.sev-mild     { background:rgba(180,220,0,0.12); border:1px solid rgba(180,220,0,0.35); color:#b4dc00; }
.sev-moderate { background:rgba(255,160,0,0.12); border:1px solid rgba(255,160,0,0.35); color:#ffa000; }
.sev-severe   { background:rgba(255,60,60,0.12); border:1px solid rgba(255,60,60,0.35); color:#ff3c3c; }
.sev-dot { width:8px; height:8px; border-radius:50%; background:currentColor; animation:dotBlink 1.4s ease-in-out infinite; }
@keyframes dotBlink { 0%,100%{opacity:1;} 50%{opacity:0.2;} }

/* ── Progress bar ─────────────────────────────────────────── */
.prog-wrap { margin: 1rem 0; }
.prog-label {
  display: flex; justify-content: space-between;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem; color: var(--muted); margin-bottom: 6px;
}
.prog-track {
  background: rgba(255,255,255,0.05);
  border-radius: 100px;
  height: 8px;
  overflow: hidden;
  border: 1px solid var(--glass-border);
}
.prog-fill {
  height: 100%;
  border-radius: 100px;
  background: linear-gradient(90deg, #00c850, #00e87a, #39ff9e);
  box-shadow: 0 0 12px rgba(0,232,122,0.5);
  transition: width 1.2s cubic-bezier(0.23,1,0.32,1);
  animation: fillGlow 2s ease-in-out infinite alternate;
}
@keyframes fillGlow {
  0%   { box-shadow: 0 0 8px rgba(0,232,122,0.3); }
  100% { box-shadow: 0 0 20px rgba(0,232,122,0.7); }
}

/* ── Disease expanders ────────────────────────────────────── */
[data-testid="stExpander"] {
  background: linear-gradient(135deg, rgba(10,40,20,0.9), rgba(5,18,10,0.95)) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 14px !important;
  margin-bottom: 12px !important;
  transition: all 0.35s ease !important;
  overflow: hidden;
}
[data-testid="stExpander"]:hover {
  border-color: var(--accent) !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(0,232,122,0.1) !important;
}
[data-testid="stExpander"] summary {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: var(--accent) !important;
  letter-spacing: 0.08em;
  padding: 14px 18px !important;
}
[data-testid="stExpanderDetails"] { padding: 0 18px 18px !important; }
[data-testid="stExpanderDetails"] p { color: var(--text) !important; font-size: 0.9rem; line-height: 1.65; }
[data-testid="stExpanderDetails"] strong { color: var(--accent) !important; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.08em; }

/* ── Download button ──────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
  width: 100%;
  background: linear-gradient(135deg, rgba(0,232,122,0.12), rgba(0,180,90,0.16)) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 14px !important;
  color: var(--accent) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  padding: 14px 24px !important;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: all 0.35s ease !important;
}
[data-testid="stDownloadButton"] > button::before {
  content: '';
  position: absolute;
  top: 50%; left: -100%;
  width: 100%; height: 200%;
  background: linear-gradient(90deg, transparent, rgba(0,232,122,0.15), transparent);
  transform: translateY(-50%) skewX(-20deg);
  transition: left 0.5s ease;
}
[data-testid="stDownloadButton"] > button:hover { 
  background: linear-gradient(135deg, rgba(0,232,122,0.22), rgba(0,200,100,0.26)) !important;
  box-shadow: 0 0 30px rgba(0,232,122,0.3), 0 8px 24px rgba(0,0,0,0.4) !important;
  transform: translateY(-2px) !important;
}
[data-testid="stDownloadButton"] > button:hover::before { left: 100%; }

/* ── Spinner ──────────────────────────────────────────────── */
[data-testid="stSpinner"] * { color: var(--accent) !important; }

/* ── Divider ──────────────────────────────────────────────── */
hr { border-color: var(--glass-border) !important; margin: 2rem 0 !important; }

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--g1); }
::-webkit-scrollbar-thumb { background: var(--g3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Idle screen ──────────────────────────────────────────── */
.idle-wrap {
  min-height: 65vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 4rem 2rem;
}
.idle-icon {
  font-size: 5rem;
  display: block;
  margin-bottom: 1.5rem;
  animation: idleFloat 4s ease-in-out infinite;
  filter: drop-shadow(0 0 28px rgba(0,232,122,0.5));
}
@keyframes idleFloat {
  0%, 100% { transform: translateY(0) rotate(-5deg); }
  50%       { transform: translateY(-20px) rotate(5deg); }
}
.idle-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.4rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent), var(--accent3));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.8rem;
}
.idle-sub {
  color: var(--muted);
  font-size: 1rem;
  max-width: 480px;
  line-height: 1.7;
  margin-bottom: 2rem;
}
.idle-steps {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 1rem;
}
.idle-step {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  padding: 1rem 1.4rem;
  width: 180px;
  transition: all 0.3s ease;
}
.idle-step:hover {
  border-color: var(--accent);
  transform: translateY(-4px);
  box-shadow: 0 12px 28px rgba(0,0,0,0.4);
}
.idle-step .is-num {
  font-family: 'Playfair Display', serif;
  font-size: 2rem;
  color: var(--accent);
  line-height: 1;
  margin-bottom: 6px;
}
.idle-step .is-txt {
  font-size: 0.8rem;
  color: var(--muted);
  line-height: 1.5;
}

/* ── Tag chips ────────────────────────────────────────────── */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.chip {
  background: rgba(0,232,122,0.08);
  border: 1px solid var(--glass-border);
  border-radius: 100px;
  padding: 3px 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--accent);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ── Streamlit overrides ──────────────────────────────────── */
.stMarkdown { color: var(--text) !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family:'JetBrains Mono',monospace !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-family:'JetBrains Mono',monospace !important; font-size:0.7rem !important; text-transform:uppercase; letter-spacing:0.1em; }
[data-testid="stMetricDelta"] > div { color: var(--accent2) !important; }
</style>

<!-- Floating leaf particles -->
<div class="leaf-field">
  <div class="leaf"></div><div class="leaf"></div><div class="leaf"></div>
  <div class="leaf"></div><div class="leaf"></div><div class="leaf"></div>
  <div class="leaf"></div><div class="leaf"></div><div class="leaf"></div>
  <div class="leaf"></div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <h1 class="hero-title">Multi-Disease Detection &amp; Severity Grading in Guava Leaves</h1>
  <p class="hero-sub">using YOLOv8 &amp; Grad-CAM Explainability</p>
  <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <span class="sb-icon">🍃</span>
      <div class="sb-title">GuavaGuard AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-strip">
      <div class="stat-pill">
        <span class="sp-val">5</span>
        <span class="sp-lbl">Classes</span>
      </div>
      <div class="stat-pill">
        <span class="sp-val">YOLOv8</span>
        <span class="sp-lbl">Model</span>
      </div>
      <div class="stat-pill">
        <span class="sp-val">XAI</span>
        <span class="sp-lbl">Explainable</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--accent);margin-bottom:8px;">⚙️ Analysis Mode</p>', unsafe_allow_html=True)
    mode = st.radio(
        "SENSING MODE",
        ["🔬 Standard Diagnostic", "🧠 Enhanced XAI (Grad-CAM)"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--accent);margin-bottom:8px;">📥 Data Ingestion</p>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload leaf images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

# ──────────────────────────────────────────────────────────────────
# SEVERITY HELPER
# ──────────────────────────────────────────────────────────────────
# Map the severity level (from inference.classify) to its badge CSS class,
# so the badge colour always matches the reported level.
LEVEL_CSS = {
    "Healthy":  "sev-healthy",
    "Mild":     "sev-mild",
    "Moderate": "sev-moderate",
    "Severe":   "sev-severe",
}


def severity_class(level: str) -> str:
    return LEVEL_CSS.get(level, "sev-healthy")


# ──────────────────────────────────────────────────────────────────
# MAIN — FILE PROCESSING
# ──────────────────────────────────────────────────────────────────
if files:
    for idx, file in enumerate(files):
        img_bytes = np.frombuffer(file.getvalue(), np.uint8)
        img = cv2.imdecode(img_bytes, 1)

        col_viz, col_anal = st.columns([1.55, 1], gap="large")

        with st.spinner("⚡ Conducting molecular scan…"):
            yolo_img, sev, level, diseases, results, boxes = run_yolo(img)
            if "Grad-CAM" in mode:
                final_view = run_gradcam(img, boxes)
            else:
                final_view = yolo_img

        # ── LEFT: Visualisation ──────────────────────────────
        with col_viz:
            st.markdown('<div class="sec-heading">🛰️ Diagnostic Overlay</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="scan-shell">
              <div class="scan-line-anim"></div>
              <div class="corner tl"></div>
              <div class="corner tr"></div>
              <div class="corner bl"></div>
              <div class="corner br"></div>
            """, unsafe_allow_html=True)
            st.image(cv2.cvtColor(final_view, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # file info chips
            st.markdown(f"""
            <div class="chip-row" style="margin-top:14px;">
              <span class="chip">📄 {file.name}</span>
              <span class="chip">{'🧠 Grad-CAM' if 'Grad-CAM' in mode else '🔬 Standard'}</span>
              <span class="chip">🕒 {datetime.now().strftime('%H:%M:%S')}</span>
            </div>
            """, unsafe_allow_html=True)

        # ── RIGHT: Analytics ──────────────────────────────────
        with col_anal:
            st.markdown('<div class="sec-heading">📊 Pathology Analytics</div>', unsafe_allow_html=True)

            # Metric cards
            sev_cls = severity_class(level)
            st.markdown(f"""
            <div class="metric-row">
              <div class="m-card">
                <div class="mc-label">Pathogen Load</div>
                <div class="mc-value">{sev:.1f}<span style="font-size:1rem">%</span></div>
                <div class="mc-delta">Leaf area affected</div>
              </div>
              <div class="m-card">
                <div class="mc-label">Diseases</div>
                <div class="mc-value">{len(diseases)}</div>
                <div class="mc-delta">Detected classes</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Severity badge
            st.markdown(f"""
            <div style="margin-bottom:1.2rem;">
              <div class="mc-label" style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--muted);margin-bottom:8px;">Severity Assessment</div>
              <span class="sev-badge {sev_cls}">
                <span class="sev-dot"></span>
                {level}
              </span>
            </div>
            """, unsafe_allow_html=True)

            # Severity progress bar
            st.markdown(f"""
            <div class="prog-wrap">
              <div class="prog-label">
                <span>INFECTION INDEX</span>
                <span>{sev:.1f}%</span>
              </div>
              <div class="prog-track">
                <div class="prog-fill" style="width:{min(sev,100):.1f}%;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Download
            pdf_data = generate_detailed_pdf(diseases, sev, level, final_view)
            st.download_button(
                label="📥  Download Full Diagnostic Report (PDF)",
                data=pdf_data,
                file_name=f"GuavaGuard_Report_{file.name.rsplit('.',1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                key=f"download_btn_{idx}_{file.name}",
                use_container_width=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Disease expanders
            if diseases:
                st.markdown('<div class="sec-heading">🔴 Detected Pathologies</div>', unsafe_allow_html=True)
                for d in diseases:
                    info = disease_info.get(d, {})
                    with st.expander(f"⬤  {d.replace('_',' ').upper()}", expanded=True):
                        for label, field in [
                            ("What it is",          "description"),
                            ("Causal agent",         "cause"),
                            ("Damage caused",        "impact"),
                            ("Immediate treatment",  "treatment"),
                            ("Organic option",       "organic"),
                            ("Prevention",           "prevention"),
                        ]:
                            st.markdown(f"**{label}:** {info.get(field,'N/A')}")
                        st.markdown(
                            f"<p style='color:var(--accent2);margin-top:8px;'>"
                            f"<strong>🌱 Future-ready:</strong> {info.get('future_safety','N/A')}"
                            f"</p>",
                            unsafe_allow_html=True
                        )
            else:
                st.markdown("""
                <div style="text-align:center;padding:2rem;background:var(--glass);border:1px solid var(--glass-border);border-radius:16px;">
                  <span style="font-size:2.5rem;">✅</span>
                  <p style="color:var(--accent);font-family:'JetBrains Mono',monospace;font-size:0.85rem;margin-top:8px;letter-spacing:0.1em;">NO PATHOGENS DETECTED</p>
                  <p style="color:var(--muted);font-size:0.85rem;">Leaf appears healthy</p>
                </div>
                """, unsafe_allow_html=True)

        # Divider
        st.markdown("""<hr>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# IDLE SCREEN
# ──────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="idle-wrap">
      <span class="idle-icon">🍃</span>
      <p class="idle-sub">
        Upload a guava leaf image from the sidebar to begin AI-powered
        multi-disease detection and severity grading analysis.
      </p>
      <div class="idle-steps">
        <div class="idle-step">
          <div class="is-num">01</div>
          <div class="is-txt">Select analysis mode from the sidebar</div>
        </div>
        <div class="idle-step">
          <div class="is-num">02</div>
          <div class="is-txt">Upload one or more guava leaf images</div>
        </div>
        <div class="idle-step">
          <div class="is-num">03</div>
          <div class="is-txt">Receive instant AI diagnostics & PDF report</div>
        </div>
      </div>
      <div class="chip-row" style="margin-top:2.5rem;justify-content:center;">
        <span class="chip">YOLOv8 Detection</span>
        <span class="chip">Grad-CAM XAI</span>
        <span class="chip">Severity Grading</span>
        <span class="chip">PDF Reports</span>
        <span class="chip">Multi-Disease</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
