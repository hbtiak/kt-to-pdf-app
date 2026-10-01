"""
KT Recording → PDF Generator
Upload a video file, transcribe it, and generate a downloadable PDF.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------
# CRITICAL FIX: Clear executable stack flag from ctranslate2 library
# This must run BEFORE any import of faster_whisper or ctranslate2.
# ------------------------------------------------------------------
def _fix_ctranslate2_execstack():
    """Finds the ctranslate2 shared library and clears its execstack flag."""
    try:
        # Find the site-packages directory for the current Python environment
        import site
        site_packages = site.getsitepackages()[0]
        
        # Find the library file matching the pattern
        lib_pattern = "libctranslate2-*.so.*"
        ctranslate2_lib_dir = Path(site_packages) / "ctranslate2.libs"
        
        if not ctranslate2_lib_dir.exists():
            return  # Library directory not found, skip fix

        for lib_file in ctranslate2_lib_dir.glob(lib_pattern):
            print(f"Applying execstack fix to: {lib_file}")
            # Use patchelf to clear the executable stack flag
            result = subprocess.run(
                ["patchelf", "--clear-execstack", str(lib_file)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"patchelf warning: {result.stderr}")
            else:
                print(f"Successfully cleared execstack for {lib_file.name}")
    except Exception as e:
        # Log the error but don't crash the app
        print(f"Could not apply ctranslate2 execstack fix: {e}")

# Run the fix immediately
_fix_ctranslate2_execstack()

# ------------------------------------------------------------------
# Now, safe to import everything else
# ------------------------------------------------------------------
import streamlit as st
from fpdf import FPDF

# ------------------------------------------------------------------
# PAGE CONFIG — must be first Streamlit command
# ------------------------------------------------------------------
st.set_page_config(
    page_title="KT Recording → PDF",
    page_icon="📄",
    layout="wide",
)

st.title("📄 KT Recording → PDF Generator")
st.caption(
    "Upload a KT recording (mp4, mp3, wav, m4a, mov), and the app will "
    "transcribe it and generate a downloadable PDF."
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
        help="Larger = more accurate but slower. 'base' is good for CPU.",
    )

    include_summary = st.checkbox("Include summary page", value=True)

    st.divider()
    st.markdown("**Supported formats**")
    st.markdown(
        "- Video: `mp4`, `mov`, `mkv`, `webm`, `avi`\n"
        "- Audio: `mp3`, `wav`, `m4a`, `aac`, `ogg`"
    )

# ------------------------------------------------------------------
# UPLOAD FORM
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "🎬 Upload recording",
    type=["mp4", "mov", "mkv", "webm", "avi",
          "mp3", "wav", "m4a", "aac", "ogg"],
    accept_multiple_files=False,
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

# ------------------------------------------------------------------
# STEP 1 — SAVE UPLOADED FILE
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
# STEP 2 — EXTRACT AUDIO (if video)
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
    # This import now works because of the _fix_ctranslate2_execstack() call
    from faster_whisper import WhisperModel

    with st.spinner(f"Loading Whisper '{model_size}' model..."):
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    with st.spinner("Transcribing (this may take a while)..."):
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            word_timestamps=False,
        )
        result_segments = []
        full_text_parts = []
        for seg in segments:
            result_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
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
# STEP 4 — PDF BUILD
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

def build_pdf(transcript: dict, title: str, out_path: Path, include_summary: bool):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Cover
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 120, 212)
    pdf.ln(30)
    pdf.multi_cell(0, 12, _safe(title), align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", ln=True)
    pdf.cell(0, 8, f"Duration: {format_timestamp(transcript.get('duration', 0))}", align="C", ln=True)
    pdf.cell(0, 8, f"Language: {transcript.get('language', 'unknown')}", align="C", ln=True)

    # Summary
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

    # Full transcript
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 120, 212)
    pdf.cell(0, 10, "Full Transcript", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 30, 30)
    for seg in transcript.get("segments", []):
        ts = format_timestamp(seg.get("start", 0))
        text = _safe(seg.get("text", "").strip())
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
        status.write("💾 Saving uploaded file...")
        media_path = save_upload(uploaded_file)
        status.write(f"✅ Saved ({media_path.stat().st_size / 1e6:.1f} MB)")

        status.write("🎧 Extracting audio...")
        audio_path = extract_audio(media_path)
        status.write("✅ Audio extracted")

        status.write(f"📝 Transcribing with Whisper '{whisper_model}'...")
        transcript = transcribe(audio_path, whisper_model)
        status.write(f"✅ Transcribed {len(transcript.get('segments', []))} segments")

        status.write("📄 Building PDF...")
        title = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ")
        pdf_path = WORKDIR / "KT_Transcript.pdf"
        build_pdf(transcript, title, pdf_path, include_summary)
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

        import base64
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
