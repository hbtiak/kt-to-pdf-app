# KT Recording → PDF Generator

Streamlit app that reads a SharePoint recording via Microsoft Graph API,
transcribes it with OpenAI Whisper, and generates a downloadable PDF.

## Prerequisites

1. **Azure AD app registration** with:
   - API permission: `Sites.Read.All`, `Files.Read.All` (Application)
   - Admin consent granted
   - Client secret created

2. **Recording must be a raw `.mp4`** in a SharePoint document library
   (not a Teams/Stream DRM-protected recording).

## Run locally

```bash
pip install -r requirements.txt
# ffmpeg must be installed on the system
streamlit run app.py
