"""
KT Recording → PDF Generator
Reads a SharePoint recording via Microsoft Graph API,
transcribes it with faster-whisper, and generates a downloadable PDF.
"""

import re
import base64
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, unquote

import streamlit as st
import requests
from fpdf import FPDF

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
    "Fetches a recording from SharePoint via Microsoft Graph API, "
    "transcribes it with Whisper, and generates a downloadable PDF."
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
        help=(
            "Larger = more accurate but slower. "
            "On Streamlit Cloud (CPU only), 'base' is recommended."
        ),
    )

    include_summary = st.checkbox("Include summary page", value=True)

    st.divider()
    st.markdown("**Azure AD app requirements**")
    st.markdown(
        "- Permissions: `Sites.Read.All`, `Files.Read.All`\n"
        "- Admin consent granted\n"
        "- Client credentials (secret) enabled"
    )

    st.divider()
    st.markdown("**Privacy**")
    st.markdown("Credentials are used only for this session and never stored.")

# ------------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------------
with st.form("input_form"):
    st.subheader("🔐 Azure AD Authentication")
    col1, col2 = st.columns(2)

    with col1:
        tenant_id = st.text_input(
            "Tenant ID",
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        )
        client_id = st.text_input(
            "Client ID",
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        )

    with col2:
        client_secret = st.text_input("Client Secret", type="password")

    st.subheader("📁 SharePoint Recording")
    sharepoint_url = st.text_input(
        "Recording URL",
        placeholder=(
            "https://yourtenant.sharepoint.com/sites/YourSite/"
            "Shared Documents/Recordings/file.mp4"
        ),
    )

    st.caption(
        "Provide the full SharePoint URL to the recording file. "
        "The app will resolve the site and file path automatically."
    )

    submitted = st.form_submit_button("🚀 Generate PDF", use_container_width=True)


# ------------------------------------------------------------------
# WORKDIR
# ------------------------------------------------------------------
WORKDIR = Path(tempfile.gettempdir()) / "kt_to_pdf"
WORKDIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# STEP 1 — AZURE AD TOKEN
# ------------------------------------------------------------------
def get_access_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    }
    resp = requests.post(url, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


# ------------------------------------------------------------------
# STEP 2 — PARSE SHAREPOINT URL
# ------------------------------------------------------------------
def parse_sharepoint_url(url: str) -> dict:
    """
    Extract hostname, site path, and file path from a SharePoint URL.
    Supports /sites/<name>/... and /personal/<user>/...
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    path = unquote(parsed.path)

    m = re.match(r"^/(sites|personal)/([^/]+)(/.*)?$", path)
    if not m:
        raise ValueError(
            "Could not parse SharePoint site from URL. "
            "Expected /sites/<name>/ or /personal/<user>/ in the path."
        )

    site_type, site_name, rest = m.groups()
    site_path = f"/{site_type}/{site_name}"
    file_path = (rest or "/").lstrip("/")

    return {
        "hostname": hostname,
        "site_path": site_path,
        "file_path": file_path,
    }


# ------------------------------------------------------------------
# STEP 3 — GRAPH API HELPERS
# ------------------------------------------------------------------
def get_site_id(token: str, hostname: str, site_path: str) -> str:
    url = f"https://graph.microsoft.com/v1.0/sites/{hostname}:{site_path}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["id"]


def get_drive_item(token: str, site_id: str, file_path: str) -> dict:
    url = (
        f"https://graph.microsoft.com/v1.0/sites/{site_id}"
        f"/drive/root:/{file_path}"
    )
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_file(download_url: str, out_path: Path) -> Path:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
    }
    with requests.get(
        download_url, headers=headers, stream=True, timeout=600
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = st.progress(0, text="Downloading recording...")
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    bar.progress(
                        min(downloaded / total, 1.0),
                        text=f"Downloading... {downloaded/1e6:.1f} MB",
                    )
        bar.empty()
    return out_path


# ------------------------------------------------------------------
# STEP 4 — AUDIO EXTRACTION
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
# STEP 5 — TRANSCRIPTION (faster-whisper)
# ------------------------------------------------------------------
def transcribe(audio_path: Path, model_size: str) -> dict:
    from faster_whisper import WhisperModel

    model_map = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
    }
    model_name = model_map.get(model_size, "base")

    with st.spinner(
        f"Loading Whisper '{model_name}' model "
        f"(first run downloads ~150MB)..."
    ):
        model = WhisperModel(model_name, device="cpu", compute_type="int8")

    with st.spinner("Transcribing (this may take a while)..."):
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            word_timestamps=False,
        )

        result_segments = []
        full_text_parts = []
        for seg in segments:
            result_segments.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
            )
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


def build_pdf(
    transcript: dict, title: str, out_path: Path, include_summary: bool
):
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
    if not (tenant_id and client_id and client_secret and sharepoint_url):
        st.error("Please fill in all required fields.")
        st.stop()

    status = st.status("Starting pipeline...", expanded=True)

    try:
        # 1. Authenticate
        status.write("🔐 Authenticating with Azure AD...")
        token = get_access_token(tenant_id, client_id, client_secret)
        status.write("✅ Token acquired")

        # 2. Parse URL
        status.write("🔍 Parsing SharePoint URL...")
        parsed = parse_sharepoint_url(sharepoint_url)
        status.write(f"✅ Site: {parsed['site_path']}")

        # 3. Resolve site ID
        status.write("🔍 Resolving site ID...")
        site_id = get_site_id(token, parsed["hostname"], parsed["site_path"])
        status.write("✅ Site resolved")

        # 4. Get file metadata
        status.write("📄 Fetching file metadata...")
        item = get_drive_item(token, site_id, parsed["file_path"])
        file_name = item.get("name", "recording.mp4")

        download_url = item.get("@microsoft.graph.downloadUrl")
        if not download_url:
            raise RuntimeError(
                "Graph API did not return a download URL for this file. "
                "The file may be DRM-protected, stored in an Asset Library, "
                "or a Stream/Teams recording that doesn't expose a raw MP4."
            )
        status.write(f"✅ File: {file_name}")

        # 5. Download
        status.write("⬇️ Downloading recording...")
        video_path = WORKDIR / file_name
        download_file(download_url, video_path)
        status.write(
            f"✅ Downloaded {video_path.stat().st_size / 1e6:.1f} MB"
        )

        # 6. Extract audio
        status.write("🎧 Extracting audio...")
        audio_path = extract_audio(video_path)
        status.write("✅ Audio extracted")

        # 7. Transcribe
        status.write(f"📝 Transcribing with Whisper '{whisper_model}'...")
        transcript = transcribe(audio_path, whisper_model)
        status.write(
            f"✅ Transcribed {len(transcript.get('segments', []))} segments"
        )

        # 8. Build PDF
        status.write("📄 Building PDF...")
        title = Path(file_name).stem.replace("_", " ").replace("-", " ")
        pdf_path = WORKDIR / "KT_Transcript.pdf"
        build_pdf(transcript, title, pdf_path, include_summary)
        status.write("✅ PDF generated")

        status.update(label="Pipeline complete ✅", state="complete")

        # 9. Deliver
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
        st.info(
            "**Common issues:**\n"
            "- Admin consent not granted for `Sites.Read.All` / `Files.Read.All`\n"
            "- URL points to a Stream page instead of a file path\n"
            "- File is DRM-protected or in an Asset Library\n"
            "- Wrong tenant ID / client ID / client secret\n"
            "- Personal OneDrive files are not accessible with app-only auth"
        )
