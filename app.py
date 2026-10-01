import streamlit as st
import requests
import tempfile
import subprocess
from pathlib import Path
from fpdf import FPDF
import whisper

st.set_page_config(page_title="KT Recording → PDF", layout="wide")
st.title("📄 KT Recording → PDF Generator")

# ---- Sidebar ----
with st.sidebar:
    st.header("Configuration")
    tenant_id = st.text_input("Tenant ID")
    client_id = st.text_input("Client ID")
    client_secret = st.text_input("Client Secret", type="password")
    whisper_model = st.selectbox("Whisper Model", ["base", "small", "medium"], index=0)

# ---- Input ----
sharepoint_url = st.text_input(
    "SharePoint Recording URL or File Path",
    placeholder="https://yourtenant.sharepoint.com/sites/.../Recording.mp4"
)

if st.button("Generate PDF", use_container_width=True):
    if not all([tenant_id, client_id, client_secret, sharepoint_url]):
        st.error("Please fill in all fields")
        st.stop()
    
    status = st.status("Processing...", expanded=True)
    
    try:
        # 1. Get access token
        status.write("🔐 Authenticating...")
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        token_resp = requests.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default"
        })
        token_resp.raise_for_status()
        token = token_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 2. Parse SharePoint URL to get site and file path
        from urllib.parse import urlparse, unquote
        parsed = urlparse(sharepoint_url)
        hostname = parsed.hostname
        
        # Extract site path (e.g., /sites/MySite or /personal/user)
        import re
        match = re.match(r"^/(sites|personal)/([^/]+)(/.*)?", parsed.path)
        if not match:
            raise ValueError("Could not parse SharePoint site from URL")
        
        site_path = f"/{match.group(1)}/{match.group(2)}"
        file_path = unquote(match.group(3) or "/")
        
        # 3. Get site ID
        status.write("🔍 Resolving site...")
        site_url = f"https://graph.microsoft.com/v1.0/sites/{hostname}:{site_path}"
        site_resp = requests.get(site_url, headers=headers)
        site_resp.raise_for_status()
        site_id = site_resp.json()["id"]
        
        # 4. Get file metadata
        status.write("📄 Fetching file metadata...")
        item_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{file_path.lstrip('/')}"
        item_resp = requests.get(item_url, headers=headers)
        item_resp.raise_for_status()
        item_data = item_resp.json()
        
        download_url = item_data.get("@microsoft.graph.downloadUrl")
        if not download_url:
            st.error("""
            **No download URL available.** 
            
            This usually means:
            - The file is in an Asset Library (videos use stub folders)
            - The recording is DRM-protected
            - The file requires preview API instead
            
            Try using the SharePoint Preview API endpoint instead, or 
            consider Azure Video Indexer for audio extraction.
            """)
            st.stop()
        
        # 5. Download video
        status.write("⬇️ Downloading recording...")
        video_path = Path(tempfile.gettempdir()) / "recording.mp4"
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(video_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
        status.write(f"✅ Downloaded {video_path.stat().st_size / 1e6:.1f} MB")
        
        # 6. Extract audio
        status.write("🎧 Extracting audio...")
        audio_path = video_path.with_suffix(".wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(audio_path)
        ], check=True, capture_output=True)
        status.write("✅ Audio extracted")
        
        # 7. Transcribe
        status.write("📝 Transcribing...")
        model = whisper.load_model(whisper_model)
        result = model.transcribe(str(audio_path), word_timestamps=True)
        status.write("✅ Transcription complete")
        
        # 8. Build PDF
        status.write("📄 Building PDF...")
        
        class PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(120, 120, 120)
                self.cell(0, 8, "KT Session Transcript", align="R")
                self.ln(10)
            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")
        
        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Title
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(0, 120, 212)
        title = Path(item_data.get("name", "Recording")).stem.replace("_", " ")
        pdf.multi_cell(0, 10, title, align="C")
        pdf.ln(10)
        
        # Metadata
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 6, f"Duration: {result.get('duration', 0):.0f} seconds", ln=True)
        pdf.cell(0, 6, f"Language: {result.get('language', 'unknown')}", ln=True)
        pdf.ln(10)
        
        # Transcript
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 120, 212)
        pdf.cell(0, 10, "Full Transcript", ln=True)
        pdf.ln(4)
        
        for seg in result.get("segments", []):
            start = seg.get("start", 0)
            ts = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d}"
            
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(0, 120, 212)
            pdf.cell(22, 5, f"[{ts}]", ln=False)
            
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            text = seg.get("text", "").strip().encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 5, text)
            pdf.ln(1)
        
        # Output
        pdf_path = Path(tempfile.gettempdir()) / "KT_Transcript.pdf"
        pdf.output(str(pdf_path))
        
        status.update(label="Complete ✅", state="complete")
        st.success("PDF ready!")
        
        # Download button
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF",
                data=f,
                file_name=f"KT_Transcript.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
    except Exception as e:
        status.update(label="Failed ❌", state="error")
        st.exception(e)
