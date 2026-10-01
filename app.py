"""
Video → PDF Generator
Downloads a YouTube video (or any yt-dlp-supported URL),
transcribes it with faster-whisper, and generates a downloadable PDF.
"""

import os
import re
import base64
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st
from fpdf import FPDF

# ------------------------------------------------------------------
# PAGE CONFIG — MUST BE THE FIRST STREAMLIT COMMAND
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Video → PDF",
    page_icon="📄",
    layout="wide",
)

# ------------------------------------------------------------------
# ONE-TIME SETUP: Ensure Deno JS runtime for yt-dlp
# ------------------------------------------------------------------
def _ensure_deno():
    """Install Deno manually into a user-writable directory."""
    deno_dir = Path.home() / ".deno" / "bin"
    deno_bin = deno_dir / "deno"

    if deno_bin.exists():
        os.environ["PATH"] = str(deno_dir) + ":" + os.environ.get("PATH", "")
        return

    deno_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract Deno
    subprocess.run(
        f"curl -fsSL https://github.com/denoland/deno/releases/latest/download/"
        f"deno-x86_64-unknown-linux-gnu.zip -o /tmp/deno.zip",
        shell=True, check=False,
    )
    subprocess.run(
        f"unzip -o /tmp/deno.zip -d {deno_dir}",
        shell=True, check=False,
    )
    subprocess.run(f"chmod +x {deno_bin}", shell=True, check=False)

    if deno_bin.exists():
        os.environ["PATH"] = str(deno_dir) + ":" + os.environ.get("PATH", "")

# Run Deno setup (it will NOT call st.* functions, so it's safe here)
_ensure_deno()

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📄 Video → PDF Generator")
st.caption(
    "Downloads a video via yt-dlp, transcribes it with Whisper, "
    "and generates a downloadable PDF."
)

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
    st.markdown("**Supported URLs**")
    st.markdown(
        "- YouTube (`youtube.com/watch?v=...`, `youtu.be/...`)\n"
        "- Any site supported by yt-dlp\n"
        "- Public, non-age-restricted videos work best"
    )

with st.form("input_form"):
    video_url = st.text_input(
        "🎬 Video URL",
        placeholder="https://www.youtube.com/watch?v=Mu3POlNoLdc",
    )
    submitted = st.form_submit_button("🚀 Generate PDF", use_container_width=True)

# ------------------------------------------------------------------
# WORKDIR
# ------------------------------------------------------------------
WORKDIR = Path(tempfile.gettempdir()) / "video_to_pdf"
WORKDIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# STEP 1 — DOWNLOAD VIDEO
# ------------------------------------------------------------------
def download_video(url: str, out_dir: Path) -> Path:
    for f in out_dir.glob("video.*"):
        try:
            f.unlink()
        except Exception:
            pass

    out_template = str(out_dir / "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "--remote-components", "ejs:github",
        "--js-runtimes", "deno",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--no-playlist",
        "--no-warnings",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    files = list(out_dir.glob("video.*"))
    if not files:
        raise RuntimeError("yt-dlp reported success but no output file was found.")
    return files[0]

# ------------------------------------------------------------------
# STEP 2 — EXTRACT AUDIO
# ------------------------------------------------------------------
def extract_audio(video_path: Path) -> Path:
    audio_path = video_path.with_suffix(".wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
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

    with st.spinner("Transcribing (this may take a while)..."):
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
# STEP 4 — PDF BUILD
# ------------------------------------------------------------------
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Video Transcript", align="R")
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def _safe(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")

def build_pdf(transcript: dict, title: str, url: str, out_path: Path, include_summary: bool):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 120, 212)
    pdf.ln(30)
    pdf.multi_cell(0, 10, _safe(title), align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, _safe(f"Source: {url}"), align="C", ln=True)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", ln=True)
    pdf.cell(0, 8, f"Duration: {format_timestamp(transcript.get('duration', 0))}", align="C", ln=True)
    pdf.cell(0, 8, f"Language: {transcript.get('language', 'unknown')}", align="C", ln=True)

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
if submitted:
    if not video_url:
        st.error("Please provide a video URL.")
        st.stop()

    status = st.status("Starting pipeline...", expanded=True)
    try:
        status.write("⬇️ Downloading video...")
        video_path = download_video(video_url, WORKDIR)
        status.write(f"✅ Downloaded ({video_path.stat().st_size / 1e6:.1f} MB)")

        status.write("🎧 Extracting audio...")
        audio_path = extract_audio(video_path)
        status.write("✅ Audio extracted")

        status.write(f"📝 Transcribing with Whisper '{whisper_model}'...")
        transcript = transcribe(audio_path, whisper_model)
        status.write(f"✅ Transcribed {len(transcript.get('segments', []))} segments")

        status.write("📄 Building PDF...")
        title = f"Video: {video_url}"
        pdf_path = WORKDIR / "transcript.pdf"
        build_pdf(transcript, title, video_url, pdf_path, include_summary)
        status.write("✅ PDF generated")

        status.update(label="Pipeline complete ✅", state="complete")
        st.success("PDF is ready!")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
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
        st.info(
            "**Common issues:**\n"
            "- Video is age-restricted or region-blocked\n"
            "- yt-dlp is outdated\n"
            "- Deno failed to install (check build logs)\n"
            "- Live stream still in progress\n"
            "- ffmpeg missing from packages.txt"
        )
