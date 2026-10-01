"""
YouTube Video → PDF Generator
Downloads a YouTube video, transcribes it with faster-whisper,
and generates a downloadable PDF.
"""

import re
import base64
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st
from fpdf import FPDF

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Video → PDF",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Video → PDF Generator")
st.caption(
    "Downloads a YouTube video, transcribes it with Whisper, "
    "and generates a downloadable PDF."
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
        help="Larger = more accurate but slower.",
    )

    include_summary = st.checkbox("Include summary page", value=True)

    st.divider()
    st.markdown("**Supported URLs**")
    st.markdown(
        "- `https://www.youtube.com/watch?v=...`\n"
        "- `https://youtu.be/...`\n"
        "- `https://www.youtube.com/live/...`"
    )

# ------------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------------
with st.form("input_form"):
    video_url = st.text_input(
        "🎬 YouTube URL",
        placeholder="https://www.youtube.com/watch?v=Mu3POlNoLdc",
    )
    submitted = st.form_submit_button("🚀 Generate PDF", use_container_width=True)


# ------------------------------------------------------------------
# WORKDIR
# ------------------------------------------------------------------
WORKDIR = Path(tempfile.gettempdir()) / "video_to_pdf"
WORKDIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# STEP 1 — DOWNLOAD FROM YOUTUBE
# ------------------------------------------------------------------
def download_youtube(url: str, out_dir: Path) -> Path:
    """Download YouTube video with yt-dlp, return the mp4 path."""
    out_template = str(out_dir / "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    # Find the downloaded file
    files = list(out_dir.glob("video.*"))
    if not files:
        raise RuntimeError("yt-dlp did not produce an output file.")
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
# STEP 3 — TRANSCRIBE (faster-whisper)
# ------------------------------------------------------------------
def transcribe(audio_path: Path, model_size: str) -> dict:
    from faster_whisper import WhisperModel

    with st.spinner(
        f"Loading Whisper '{model_size}' model "
        f"(first run downloads the model)..."
    ):
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

    # ---- Cover ----
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 120, 212)
    pdf.ln(30)
    pdf.multi_cell(0, 10, _safe(title), align="C")

    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Source: {url}", align="C", ln=True)
    pdf.cell(
        0, 8,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        align="C", ln=True,
    )
    pdf.cell(
        0, 8,
        f"Duration: {format_timestamp(transcript.get('duration', 0))}",
        align="C", ln=True,
    )
    pdf.cell(
        0, 8,
        f"Language: {transcript.get('language', 'unknown')}",
        align="C", ln=True,
    )

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

    # ---- Full transcript ----
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
        st.error("Please provide a YouTube URL.")
        st.stop()

    status = st.status("Starting pipeline...", expanded=True)

    try:
        # 1. Download
        status.write("⬇️ Downloading video from YouTube...")
        video_path = download_youtube(video_url, WORKDIR)
        size_mb = video_path.stat().st_size / 1e6
        status.write(f"✅ Downloaded ({size_mb:.1f} MB)")

        # 2. Extract audio
        status.write("🎧 Extracting audio...")
        audio_path = extract_audio(video_path)
        status.write("✅ Audio extracted")

        # 3. Transcribe
        status.write(f"📝 Transcribing with Whisper '{whisper_model}'...")
        transcript = transcribe(audio_path, whisper_model)
        status.write(
            f"✅ Transcribed {len(transcript.get('segments', []))} segments"
        )

        # 4. Build PDF
        status.write("📄 Building PDF...")
        title = f"YouTube Video {video_url}"
        pdf_path = WORKDIR / "transcript.pdf"
        build_pdf(transcript, title, video_url, pdf_path, include_summary)
        status.write("✅ PDF generated")

        status.update(label="Pipeline complete ✅", state="complete")

        # 5. Deliver
        st.success("PDF is ready!")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download PDF",
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
            "- yt-dlp needs updating (YouTube changes frequently)\n"
            "- Video is a live stream still in progress\n"
            "- Check that ffmpeg is installed via packages.txt"
        )
