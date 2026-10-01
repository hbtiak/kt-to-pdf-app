"""
KT Recording → Structured Document PDF
Upload a video → transcript → Gemini structures it into headings/paragraphs
→ CLIP picks relevant screenshots → PDF with optional transcript appendix.
"""

import os
import json
import time
import subprocess
import tempfile
import concurrent.futures
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------
# execstack fix for ctranslate2
# ------------------------------------------------------------------
def _fix_execstack_for(package_name: str):
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
import numpy as np

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="KT Recording → Document",
    page_icon="📄",
    layout="wide",
)

st.title("📄 KT Recording → Structured Document")
st.caption(
    "Upload a recording. The app transcribes it, uses Gemini to structure it "
    "into headings and paragraphs, adds relevant screenshots, and generates a PDF."
)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("LLM (Gemini Free Tier)")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    llm_model = st.selectbox(
        "Gemini model",
        [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
        ],
        index=0,
        help=(
            "If 3.8 Flash returns 503, try 3.1 Flash-Lite — it's often "
            "more reliable during peak demand."
        ),
    )

    st.divider()
    st.subheader("Transcription")
    whisper_model = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium"],
        index=1,
    )

    st.divider()
    st.subheader("Screenshots")
    include_screenshots = st.checkbox(
        "Add AI-selected screenshots", value=True,
    )
    if include_screenshots:
        num_screenshots = st.slider("Max screenshots", 1, 10, 4)
        scene_threshold = st.slider(
            "Scene-change sensitivity", 0.10, 0.60, 0.30,
        )
    else:
        num_screenshots = 0
        scene_threshold = 0.30

    st.divider()
    st.subheader("Output")
    include_transcript = st.checkbox(
        "Include full transcript appendix", value=False,
        help="Adds the raw timestamped transcript at the end of the PDF.",
    )

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

generate = st.button("🚀 Generate Document", use_container_width=True, type="primary")

# ------------------------------------------------------------------
# WORKDIR
# ------------------------------------------------------------------
WORKDIR = Path(tempfile.gettempdir()) / "kt_to_pdf"
WORKDIR.mkdir(exist_ok=True)
FRAMES_DIR = WORKDIR / "frames"


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    out_path = WORKDIR / f"upload{suffix}"
    if out_path.exists():
        out_path.unlink()
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


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


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ------------------------------------------------------------------
# TRANSCRIBE
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


# ------------------------------------------------------------------
# GEMINI: STRUCTURE TRANSCRIPT (with timeout + retry)
# ------------------------------------------------------------------
STRUCTURE_PROMPT = """You are an expert technical writer. You will receive a raw transcript of a training/knowledge-transfer video.

Your job: convert it into a well-structured document that a reader can skim and understand.

RULES:
1. Identify 3-7 main topics covered in the transcript. These become SECTION HEADINGS.
2. For each section, write a clear 2-4 sentence paragraph in your own words summarizing what was said. Do NOT copy-paste; synthesize.
3. Under each section, add 2-4 bullet points of key facts, definitions, or examples mentioned.
4. Add a short Executive Summary (3-5 sentences) at the top.
5. Add a "Key Takeaways" section at the end (3-5 bullets).
6. Include a "timestamp" field for each section - the start time (in seconds) where that topic begins in the video. This is used to attach screenshots.

Return ONLY valid JSON, matching this schema exactly:

{
  "title": "string - concise title for the document",
  "executive_summary": "string - 3-5 sentences",
  "sections": [
    {
      "heading": "string",
      "timestamp": number,
      "paragraph": "string - 2-4 sentences",
      "bullets": ["string", "string", "..."]
    }
  ],
  "key_takeaways": ["string", "..."]
}

Do not include markdown, code fences, or any text outside the JSON.
"""


def structure_with_llm(
    transcript: dict, api_key: str, model: str, max_retries: int = 3
) -> dict:
    """
    Call Gemini (via OpenAI-compatible endpoint).
    Uses a hard timeout so hung requests trigger retry instead of blocking.
    """
    from openai import OpenAI

    # Build timestamped transcript
    lines = []
    for seg in transcript.get("segments", []):
        ts = format_timestamp(seg.get("start", 0))
        lines.append(f"[{ts}] {seg['text'].strip()}")
    transcript_with_ts = "\n".join(lines)

    if len(transcript_with_ts) > 60_000:
        transcript_with_ts = transcript_with_ts[:60_000] + "\n[...truncated...]"

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=60.0,
    )

    def _make_call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": STRUCTURE_PROMPT},
                {"role": "user", "content": transcript_with_ts},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

    last_error = None
    for attempt in range(max_retries):
        try:
            with st.spinner(
                f"Structuring with {model}... "
                f"(attempt {attempt + 1}/{max_retries})"
            ):
                # Run in a thread with a hard 90s timeout so hangs fail fast
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(_make_call)
                    try:
                        resp = future.result(timeout=90)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError(
                            "Gemini API timed out after 90s (server hang)"
                        )

            raw = resp.choices[0].message.content
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"LLM returned invalid JSON: {e}\n\nRaw: {raw[:500]}"
                )

            if "sections" not in doc or not isinstance(doc["sections"], list):
                raise RuntimeError("LLM response missing 'sections' array.")
            return doc

        except Exception as e:
            last_error = e
            error_str = str(e)
            retryable = any(x in error_str for x in [
                "503", "429", "UNAVAILABLE", "high demand",
                "timed out", "Timeout", "overloaded",
            ])
            if retryable and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                st.warning(
                    f"Gemini is busy or hanging. Waiting {wait_time}s "
                    f"before retry {attempt + 2}/{max_retries}..."
                )
                time.sleep(wait_time)
                continue
            raise

    raise last_error


# ------------------------------------------------------------------
# FRAME EXTRACTION + CLIP SCORING
# ------------------------------------------------------------------
def extract_candidate_frames(video_path: Path, threshold: float) -> list:
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


@st.cache_resource(show_spinner=False)
def load_clip():
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer, torch


def score_frames_for_sections(candidates: list, sections: list) -> list:
    """Score each candidate frame against each section heading + paragraph."""
    import torch
    from PIL import Image as _I

    model, preprocess, tokenizer, _ = load_clip()
    if not candidates or not sections:
        return []

    section_queries = []
    for sec in sections:
        heading = sec.get("heading", "").strip()
        para = sec.get("paragraph", "").strip()
        text = f"{heading}. {para}"[:250]
        section_queries.append(text if text else "training video content")

    results = []
    with torch.no_grad():
        for cand in candidates:
            try:
                image = preprocess(
                    _I.open(cand["path"]).convert("RGB")
                ).unsqueeze(0)
                img_feat = model.encode_image(image)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
            except Exception:
                continue

            tokens = tokenizer(section_queries)
            txt_feat = model.encode_text(tokens)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

            sims = (img_feat @ txt_feat.T).squeeze(0).cpu().numpy()

            for sec_idx, sim in enumerate(sims):
                results.append({
                    "section_index": sec_idx,
                    "timestamp": cand["timestamp"],
                    "path": cand["path"],
                    "score": float(sim),
                })

    return results


def pick_screenshot_per_section(
    scored: list, num_sections: int, max_total: int
) -> dict:
    """Pick best-scoring frame for each section, cap at max_total."""
    if not scored:
        return {}

    by_section = {}
    for item in scored:
        by_section.setdefault(item["section_index"], []).append(item)

    for k in by_section:
        by_section[k].sort(key=lambda x: x["score"], reverse=True)

    picks = {}
    for sec_idx, items in by_section.items():
        if items:
            picks[sec_idx] = items[0]

    if len(picks) > max_total:
        sorted_picks = sorted(
            picks.items(), key=lambda kv: kv[1]["score"], reverse=True,
        )[:max_total]
        picks = dict(sorted_picks)

    return picks


# ------------------------------------------------------------------
# PDF BUILD
# ------------------------------------------------------------------
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Knowledge Transfer Document", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _safe(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def _soft_wrap(text: str, max_word_len: int = 40) -> str:
    """Break very long words with spaces so FPDF can wrap them."""
    result = []
    for word in text.split():
        if len(word) > max_word_len:
            word = " ".join(
                word[i:i + max_word_len]
                for i in range(0, len(word), max_word_len)
            )
        result.append(word)
    return " ".join(result)


def _add_image_fitted(
    pdf: PDF, img_path: Path, max_w_mm: float = 150, max_h_mm: float = 90
):
    try:
        img = Image.open(img_path)
        w, h = img.size
        ratio = h / w
        img_w = max_w_mm
        img_h = img_w * ratio
        if img_h > max_h_mm:
            img_h = max_h_mm
            img_w = img_h / ratio
        x = (pdf.w - img_w) / 2
        pdf.image(str(img_path), x=x, w=img_w, h=img_h)
        return True
    except Exception as e:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(200, 80, 80)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 5, f"(Screenshot failed: {e})", ln=True)
        return False


def build_document_pdf(
    doc: dict,
    transcript: dict,
    screenshots_by_section: dict,
    out_path: Path,
    include_transcript: bool,
):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---------- Cover ----------
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(0, 120, 212)
    pdf.ln(35)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        0, 12, _safe(doc.get("title", "Knowledge Transfer")), align="C"
    )

    pdf.ln(15)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        0, 8,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        align="C", ln=True,
    )
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        0, 8,
        f"Source duration: {format_timestamp(transcript.get('duration', 0))}",
        align="C", ln=True,
    )
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        0, 8,
        f"Language: {transcript.get('language', 'unknown')}",
        align="C", ln=True,
    )

    # ---------- Executive Summary ----------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 120, 212)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 12, "Executive Summary", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        0, 6, _safe(_soft_wrap(doc.get("executive_summary", "")))
    )
    pdf.ln(6)

    # ---------- Sections ----------
    for idx, section in enumerate(doc.get("sections", [])):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(0, 120, 212)
        heading = section.get("heading", f"Section {idx + 1}")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 9, _safe(_soft_wrap(heading)))
        pdf.ln(1)

        ts = section.get("timestamp", 0)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(140, 140, 140)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 5, f"@ {format_timestamp(ts)}", ln=True)
        pdf.ln(3)

        shot = screenshots_by_section.get(idx)
        if shot:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(140, 140, 140)
            pdf.set_x(pdf.l_margin)
            pdf.cell(
                0, 5,
                f"Screenshot @ {format_timestamp(shot['timestamp'])} "
                f"(relevance {shot['score']:.2f})",
                ln=True,
            )
            _add_image_fitted(pdf, shot["path"], max_w_mm=150, max_h_mm=80)
            pdf.ln(5)

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            0, 6,
            _safe(_soft_wrap(section.get("paragraph", "")))
        )
        pdf.ln(4)

        bullets = section.get("bullets", [])
        if bullets:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(60, 60, 60)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 6, "Key points:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            for b in bullets:
                pdf.set_x(pdf.l_margin)  # critical fix
                pdf.multi_cell(
                    0, 5.5,
                    _safe(_soft_wrap(f"  -  {b}"))
                )
            pdf.ln(2)

    # ---------- Key Takeaways ----------
    takeaways = doc.get("key_takeaways", [])
    if takeaways:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(0, 120, 212)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 12, "Key Takeaways", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        for t in takeaways:
            pdf.set_x(pdf.l_margin)  # critical fix
            pdf.multi_cell(0, 6, _safe(_soft_wrap(f"  -  {t}")))
            pdf.ln(2)

    # ---------- Transcript Appendix ----------
    if include_transcript:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(0, 120, 212)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 12, "Appendix: Full Transcript", ln=True)
        pdf.ln(3)

        for seg in transcript.get("segments", []):
            ts = format_timestamp(seg.get("start", 0))
            text = _safe(_soft_wrap(seg.get("text", "").strip()))
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(0, 120, 212)
            pdf.set_x(pdf.l_margin)
            pdf.cell(20, 5, f"[{ts}]", ln=False)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 5.5, text)
            pdf.ln(0.5)

    pdf.output(str(out_path))


# ------------------------------------------------------------------
# ORCHESTRATION
# ------------------------------------------------------------------
if generate and uploaded_file:
    if not gemini_api_key:
        st.error("Please provide a Gemini API key in the sidebar.")
        st.stop()

    status = st.status("Starting pipeline...", expanded=True)
    try:
        # 1. Save
        status.write("💾 Saving uploaded file...")
        media_path = save_upload(uploaded_file)
        status.write(f"✅ Saved ({media_path.stat().st_size / 1e6:.1f} MB)")

        # 2. Audio
        status.write("🎧 Extracting audio...")
        audio_path = extract_audio(media_path)
        status.write("✅ Audio extracted")

        # 3. Transcribe
        status.write(f"📝 Transcribing with Whisper '{whisper_model}'...")
        transcript = transcribe(audio_path, whisper_model)
        status.write(
            f"✅ Transcribed {len(transcript.get('segments', []))} segments"
        )

        # 4. LLM structuring
        status.write(f"🤖 Structuring document with {llm_model}...")
        doc = structure_with_llm(
            transcript, gemini_api_key, llm_model, max_retries=3
        )
        status.write(
            f"✅ Generated {len(doc.get('sections', []))} sections: "
            f"{doc.get('title', 'Untitled')}"
        )

        # 5. Screenshots
        screenshots_by_section = {}
        if include_screenshots:
            is_video = media_path.suffix.lower() in [
                ".mp4", ".mov", ".mkv", ".webm", ".avi"
            ]
            if not is_video:
                status.write("ℹ️ Audio-only upload; skipping screenshots.")
            else:
                status.write("🎞️ Detecting scene changes...")
                candidates = extract_candidate_frames(
                    media_path, scene_threshold
                )
                status.write(f"✅ Found {len(candidates)} candidate frames")

                if candidates:
                    status.write("🎯 Scoring frames against sections...")
                    scored = score_frames_for_sections(
                        candidates, doc.get("sections", [])
                    )
                    screenshots_by_section = pick_screenshot_per_section(
                        scored,
                        num_sections=len(doc.get("sections", [])),
                        max_total=num_screenshots,
                    )
                    status.write(
                        f"✅ Picked screenshots for "
                        f"{len(screenshots_by_section)} sections"
                    )

                    if screenshots_by_section:
                        st.subheader("🎯 Screenshots Attached to Sections")
                        for sec_idx, shot in sorted(
                            screenshots_by_section.items()
                        ):
                            heading = doc["sections"][sec_idx].get(
                                "heading", f"Section {sec_idx + 1}"
                            )
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.image(
                                    str(shot["path"]),
                                    caption=(
                                        f"@ {format_timestamp(shot['timestamp'])} "
                                        f"(score {shot['score']:.2f})"
                                    ),
                                    use_column_width=True,
                                )
                            with col2:
                                st.markdown(f"**{heading}**")
                                st.caption(
                                    doc["sections"][sec_idx].get(
                                        "paragraph", ""
                                    )[:200] + "..."
                                )

        # 6. Build PDF
        status.write("📄 Building document PDF...")
        pdf_path = WORKDIR / "KT_Document.pdf"
        build_document_pdf(
            doc, transcript, screenshots_by_section, pdf_path,
            include_transcript,
        )
        status.write("✅ PDF generated")

        status.update(label="Pipeline complete ✅", state="complete")
        st.success("Document ready!")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=(
                f"KT_Document_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )

        st.subheader("Preview")
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="800" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        status.update(label="Pipeline failed ❌", state="error")
        st.exception(e)
elif generate and not uploaded_file:
    st.error("Please upload a file first.")
