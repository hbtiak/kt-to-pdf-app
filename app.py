"""
KT Recording → PDF Generator (with AI-driven screenshot selection)
Upload a video, transcribe it, pick relevant frames via CLIP,
and generate a PDF with text + visuals.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------
# CRITICAL FIX: Clear executable stack flag from ctranslate2 library
# ------------------------------------------------------------------
def _fix_execstack_for(package_name: str):
    """Clear execstack flag on .so files inside <package>.libs."""
    try:
        import site
        site_packages = Path(site.getsitepackages()[0])
        libs_dir = site_packages / f"{package_name}.libs"
        if not libs_dir.exists():
            return
        for lib_file in libs_dir.glob("*.so*"):
            subprocess.run(
                ["patchelf", "--clear-execstack", str(lib_file)],
                capture_output=True, text=True,
            )
    except Exception as e:
        print(f"execstack fix skipped for {package_name}: {e}")

_fix_execstack_for("ctranslate2")

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import base64
import streamlit as st
from fpdf import FPDF
from PIL import Image

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="KT Recording → PDF",
    page_icon="📄",
    layout="wide",
)

st.title("📄 KT Recording → PDF Generator")
st.caption(
    "Upload a recording. The app transcribes it, uses CLIP to pick "
    "the most relevant frames, and generates a PDF with text + visuals."
)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    whisper_model = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium"],
        index=1,
    )

    include_summary = st.checkbox("Include summary page", value=True)
    include_screenshots = st.checkbox(
        "Add AI-selected screenshots", value=True,
        help="Uses CLIP to score frames against transcript context.",
    )

    if include_screenshots:
        num_screenshots = st.slider(
            "Max screenshots", 1, 10, 5,
            help="Upper limit on how many frames to embed.",
        )
        scene_threshold = st.slider(
            "Scene-change sensitivity", 0.10, 0.60, 0.30,
            help="Lower = more frames considered. Higher = only big changes.",
        )
    else:
        num_screenshots = 0
        scene_threshold = 0.30

# ------------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "🎬 Upload recording",
    type=["mp4", "mov", "mkv", "webm", "avi",
          "mp3", "wav", "m4a", "aac", "ogg"],
)

if uploaded_file:
    st.success(
        f"Uploaded: **{uploaded_file.name}** "
        f"({uploaded_file.size / 1e6:.1f} MB)"
    )

generate = st.button("🚀 Generate PDF", use_container_width=True, type="primary")

# ------------------------------------------------------------------
# WORKDIR
# ------------------------------------------------------------------
WORKDIR = Path(tempfile.gettempdir()) / "kt_to_pdf"
WORKDIR.mkdir(exist_ok=True)
FRAMES_DIR = WORKDIR / "frames"


# ------------------------------------------------------------------
# STEP 1 — SAVE UPLOAD
# ------------------------------------------------------------------
def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    out_path = WORKDIR / f"upload{suffix}"
    if out_path.exists():
        out_path.unlink()
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


# ------------------------------------------------------------------
# STEP 2 — EXTRACT AUDIO
# ------------------------------------------------------------------
def extract_audio(media_path: Path) -> Path:
    audio_path = WORKDIR / "audio.wav"
    if audio_path.exists():
        audio_path.unlink()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(media_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return audio_path


# ------------------------------------------------------------------
# STEP 3 — TRANSCRIBE
# ------------------------------------------------------------------
def transcribe(audio_path: Path, model_size: str) -> dict:
    from faster_whisper import WhisperModel

    with st.spinner(f"Loading Whisper '{model_size}' model..."):
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    with st.spinner("Transcribing..."):
        segments, info = model.transcribe(
            str(audio_path), beam_size=5, word_timestamps=False,
        )
        result_segments = []
        full_text_parts = []
        for seg in segments:
            result_segments.append({
                "start": seg.start, "end": seg.end, "text": seg.text,
            })
            full_text_parts.append(seg.text)

    return {
        "segments": result_segments,
        "text": "".join(full_text_parts),
        "language": info.language,
        "duration": info.duration,
    }


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ------------------------------------------------------------------
# STEP 4 — EXTRACT CANDIDATE FRAMES (scene detection)
# ------------------------------------------------------------------
def extract_candidate_frames(video_path: Path, threshold: float) -> list:
    """
    Use ffmpeg scene-change filter to get candidate frames.
    Returns list of {timestamp, path}.
    """
    if FRAMES_DIR.exists():
        for f in FRAMES_DIR.glob("*.jpg"):
            f.unlink()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    out_pattern = str(FRAMES_DIR / "frame_%04d.jpg")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-frame_pts", "1",
        out_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    timestamps = []
    for line in result.stderr.splitlines():
        if "pts_time:" in line:
            try:
                t = line.split("pts_time:")[1].split()[0]
                timestamps.append(float(t))
            except Exception:
                continue

    frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    candidates = []
    for i, frame in enumerate(frames):
        ts = timestamps[i] if i < len(timestamps) else 0.0
        candidates.append({"timestamp": ts, "path": frame})

    return candidates


# ------------------------------------------------------------------
# STEP 5 — AI SCORING (CLIP)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_clip():
    """Load CLIP once and cache it across sessions."""
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer, torch


def score_frames_with_clip(candidates: list, transcript: dict) -> list:
    """
    For each candidate frame, find the transcript window around its
    timestamp, then score image-vs-text similarity with CLIP.
    """
    import torch

    model, preprocess, tokenizer, _ = load_clip()

    segments = transcript.get("segments", [])
    if not segments or not candidates:
        return []

    def text_around(ts: float, window: float = 10.0) -> str:
        chunks = [
            s["text"] for s in segments
            if s["start"] <= ts + window and s["end"] >= ts - window
        ]
        text = " ".join(chunks).strip()
        return text if text else transcript.get("text", "")[:300]

    results = []
    with torch.no_grad():
        for cand in candidates:
            text = text_around(cand["timestamp"])
            if not text:
                continue

            image = preprocess(Image.open(cand["path"]).convert("RGB")).unsqueeze(0)
            tokens = tokenizer([text])

            img_feat = model.encode_image(image)
            txt_feat = model.encode_text(tokens)

            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

            score = float((img_feat @ txt_feat.T).item())

            results.append({
                "timestamp": cand["timestamp"],
                "path": cand["path"],
                "score": score,
                "context": text,
            })

    return results


def select_best_frames(scored: list, max_n: int) -> list:
    """Pick top-N frames, spread over time."""
    if not scored:
        return []

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    selected = []
    min_gap = 30.0

    for cand in scored:
        if len(selected) >= max_n:
            break
        if all(abs(cand["timestamp"] - s["timestamp"]) >= min_gap for s in selected):
            selected.append(cand)

    if len(selected) < max_n:
        for cand in scored:
            if cand not in selected:
                selected.append(cand)
            if len(selected) >= max_n:
                break

    return sorted(selected, key=lambda x: x["timestamp"])


# ------------------------------------------------------------------
# STEP 6 — PDF BUILD
# ------------------------------------------------------------------
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "KT Session Transcript", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _safe(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(transcript, title, out_path, include_summary, screenshots):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Cover ----
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 120, 212)
    pdf.ln(30)
    pdf.multi_cell(0, 12, _safe(title), align="C")

    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             align="C", ln=True)
    pdf.cell(0, 8, f"Duration: {format_timestamp(transcript.get('duration', 0))}",
             align="C", ln=True)
    pdf.cell(0, 8, f"Language: {transcript.get('language', 'unknown')}",
             align="C", ln=True)

    # ---- Summary ----
    if include_summary:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(0, 120, 212)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(40, 40, 40)
        segments = transcript.get("segments", [])
        picks = []
        if segments:
            picks.append(segments[0])
            if len(segments) > 2:
                picks.append(segments[len(segments) // 2])
            picks.append(segments[-1])
        for seg in picks:
            ts = format_timestamp(seg.get("start", 0))
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"[{ts}]", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, _safe(seg.get("text", "").strip()))
            pdf.ln(2)

    # ---- Full transcript with screenshots ----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 120, 212)
    pdf.cell(0, 10, "Full Transcript", ln=True)
    pdf.ln(4)

    segments = transcript.get("segments", [])

    # Map: segment id → screenshot
    shots_by_seg = {}
    if screenshots and segments:
        for shot in screenshots:
            nearest = min(
                segments,
                key=lambda s: abs(s["start"] - shot["timestamp"]),
            )
            shots_by_seg[id(nearest)] = shot

    for seg in segments:
        ts = format_timestamp(seg.get("start", 0))
        text = _safe(seg.get("text", "").strip())

        shot = shots_by_seg.get(id(seg))
        if shot:
            try:
                img = Image.open(shot["path"])
                max_w_mm = 160
                w, h = img.size
                ratio = h / w
                img_w = max_w_mm
                img_h = img_w * ratio
                if img_h > 90:
                    img_h = 90
                    img_w = img_h / ratio

                pdf.ln(2)
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(120, 120, 120)
                pdf.cell(
                    0, 5,
                    f"Screenshot @ {format_timestamp(shot['timestamp'])} "
                    f"(relevance {shot['score']:.2f})",
                    ln=True,
                )
                x = (pdf.w - img_w) / 2
                pdf.image(str(shot["path"]), x=x, w=img_w, h=img_h)
                pdf.ln(3)
            except Exception as e:
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(200, 80, 80)
                pdf.cell(0, 5, f"(Screenshot failed: {e})", ln=True)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(0, 120, 212)
        pdf.cell(22, 6, f"[{ts}]", ln=False)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 6, text)
        pdf.ln(1)

    pdf.output(str(out_path))


# ------------------------------------------------------------------
# ORCHESTRATION
# ------------------------------------------------------------------
if generate and uploaded_file:
    status = st.status("Starting pipeline...", expanded=True)
    try:
        # 1. Save
        status.write("💾 Saving uploaded file...")
        media_path = save_upload(uploaded_file)
        status.write(f"✅ Saved ({media_path.stat().st_size / 1e6:.1f} MB)")

        # 2. Extract audio
        status.write("🎧 Extracting audio...")
        audio_path = extract_audio(media_path)
        status.write("✅ Audio extracted")

        # 3. Transcribe
        status.write(f"📝 Transcribing with Whisper '{whisper_model}'...")
        transcript = transcribe(audio_path, whisper_model)
        status.write(f"✅ Transcribed {len(transcript.get('segments', []))} segments")

        # 4. Screenshots
        screenshots = []
        if include_screenshots:
            is_video = media_path.suffix.lower() in [
                ".mp4", ".mov", ".mkv", ".webm", ".avi"
            ]
            if not is_video:
                status.write("ℹ️ Uploaded file is audio-only; skipping screenshots.")
            else:
                status.write("🎞️ Detecting scene changes...")
                candidates = extract_candidate_frames(media_path, scene_threshold)
                status.write(f"✅ Found {len(candidates)} candidate frames")

                if candidates:
                    status.write("🤖 Scoring frames with CLIP (AI)...")
                    scored = score_frames_with_clip(candidates, transcript)
                    screenshots = select_best_frames(scored, num_screenshots)
                    status.write(
                        f"✅ Selected {len(screenshots)} relevant screenshots"
                    )

                    if screenshots:
                        st.subheader("🎯 AI-Selected Screenshots")
                        cols = st.columns(min(len(screenshots), 3))
                        for i, s in enumerate(screenshots):
                            with cols[i % 3]:
                                st.image(
                                    str(s["path"]),
                                    caption=(
                                        f"{format_timestamp(s['timestamp'])} "
                                        f"(score {s['score']:.2f})"
                                    ),
                                    use_column_width=True,
                                )

        # 5. Build PDF
        status.write("📄 Building PDF...")
        title = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ")
        pdf_path = WORKDIR / "KT_Transcript.pdf"
        build_pdf(transcript, title, pdf_path, include_summary, screenshots)
        status.write("✅ PDF generated")

        status.update(label="Pipeline complete ✅", state="complete")
        st.success("PDF is ready!")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"KT_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.subheader("Preview")
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="700" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        status.update(label="Pipeline failed ❌", state="error")
        st.exception(e)
elif generate and not uploaded_file:
    st.error("Please upload a file first.")
