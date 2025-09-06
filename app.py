import os
import re
import uuid
import shutil
import traceback
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydub import AudioSegment
import webrtcvad

print("Starting FastAPI backend...")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="MS-Video2Script + Video2SRT Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_coop_coep_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response

# -----------------------
# Serve React static build
# -----------------------
build_path = "build"
if not os.path.exists(build_path):
    raise RuntimeError(f"React build folder not found at '{build_path}'")

app.mount("/assets", StaticFiles(directory=os.path.join(build_path, "assets")), name="assets")

# -----------------------
# Health checks
# -----------------------
@app.get("/health")
def health():
    return {"message": "API is running ✅"}

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(build_path, "index.html"))

# -----------------------
# Upload folder
# -----------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------
# Whisper model (lazy load)
# -----------------------
model = None

def get_model():
    global model
    if model is None:
        print("Loading Whisper tiny model...")
        from faster_whisper import WhisperModel
        try:
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading Whisper model:", e)
            raise e
    return model

# -----------------------
# Helpers
# -----------------------
def seconds_to_hms(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# -----------------------
# WebRTC VAD functions
# -----------------------
ffmpeg_path = "ffmpeg"  # ensure ffmpeg is in PATH

def extract_audio_ffmpeg(video_path, audio_path="audio.wav"):
    cmd = [ffmpeg_path, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
           '-ar', '16000', '-ac', '1', audio_path]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {process.stderr.decode()}")
    return audio_path

def detect_first_speech_offset_webrtcvad(audio_path, aggressiveness=3,
                                         min_speech_ms=1000, ignore_before_ms=1000):
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    raw_audio = audio.raw_data
    vad = webrtcvad.Vad(aggressiveness)
    frame_duration = 30  # ms
    frame_size = int(16000 * frame_duration / 1000) * 2
    frames = [raw_audio[i:i+frame_size] for i in range(0, len(raw_audio), frame_size)]

    speech_start_frame = None
    speech_frames_count = 0
    required_speech_frames = min_speech_ms // frame_duration
    ignore_before_frames = ignore_before_ms // frame_duration

    for i, frame in enumerate(frames):
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate=16000):
            if speech_start_frame is None:
                speech_start_frame = i
            speech_frames_count += 1
        else:
            if speech_start_frame is not None and speech_frames_count >= required_speech_frames:
                if speech_start_frame >= ignore_before_frames:
                    return (speech_start_frame * frame_duration) / 1000.0
            speech_start_frame = None
            speech_frames_count = 0

    if speech_start_frame is not None and speech_frames_count >= required_speech_frames:
        if speech_start_frame >= ignore_before_frames:
            return (speech_start_frame * frame_duration) / 1000.0

    return 0.0

def detect_first_speech(video_path):
    audio_path = "temp.wav"
    try:
        audio_path = extract_audio_ffmpeg(video_path, audio_path)
        offset_sec = detect_first_speech_offset_webrtcvad(audio_path)
        return offset_sec
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# -----------------------
# Text splitting (for transcript only)
# -----------------------
def split_segment_text_precise(text, start_time, end_time):
    sentences = re.split(r'([.!?])', text)
    sentences = [
        (sentences[i] + (sentences[i+1] if i+1 < len(sentences) else ""))
        for i in range(0, len(sentences), 2)
    ]
    sentences = [s.strip() for s in sentences if s.strip()]
    total_chars = sum(len(s) for s in sentences)
    results = []
    current_time = start_time

    for s in sentences:
        proportion = len(s) / total_chars if total_chars > 0 else 0
        duration = proportion * (end_time - start_time)
        results.append({
            "text": s,
            "start": current_time,
            "end": current_time + duration
        })
        current_time += duration

    return results

# -----------------------
# Endpoint: Transcript (JSON)
# -----------------------
@app.post("/transcribe")
async def transcribe_text(
    video: UploadFile = File(...),
    with_timestamps: str = Form("0"),
    split_segments: str = Form("0")
):
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        whisper_model = get_model()
        first_speech_time = detect_first_speech(save_path)

        segments, info = whisper_model.transcribe(
            save_path,
            beam_size=5,
            initial_prompt=None,
            condition_on_previous_text=False,
        )

        transcription = []
        first_adjusted = False

        for seg in segments:
            if split_segments == "1":
                subs = split_segment_text_precise(seg.text, seg.start, seg.end)
                for sub in subs:
                    entry = {"text": sub["text"].strip()}
                    s_start, s_end = sub["start"], sub["end"]
                    if not first_adjusted and first_speech_time > 0:
                        s_start = first_speech_time
                        first_adjusted = True
                    if with_timestamps == "1":
                        entry["start"] = seconds_to_hms(s_start)
                        entry["end"] = seconds_to_hms(s_end)
                    transcription.append(entry)
            else:
                entry = {"text": seg.text.strip()}
                s_start, s_end = seg.start, seg.end
                if not first_adjusted and first_speech_time > 0:
                    s_start = first_speech_time
                    first_adjusted = True
                if with_timestamps == "1":
                    entry["start"] = seconds_to_hms(s_start)
                    entry["end"] = seconds_to_hms(s_end)
                transcription.append(entry)

        return {"transcription": transcription}

    except Exception as e:
        print("Error during transcription:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process transcription: {str(e)}")

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

# -----------------------
# Endpoint: Subtitles (SRT)
# -----------------------

def split_sentences(text: str):
    """Split text into sentences using punctuation."""
    parts = re.split(r'([.!?])', text)
    sentences = [
        (parts[i] + (parts[i+1] if i+1 < len(parts) else "")).strip()
        for i in range(0, len(parts), 2)
    ]
    return [s for s in sentences if s]


def wrap_text_to_two_lines(sentence: str, max_chars_per_line: int = 40):
    """
    Wrap a long sentence into max 2 lines.
    If >2 lines, merge extra into the 2nd line.
    """
    words = sentence.split()
    current_line = ""
    lines = []

    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars_per_line:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]

    return lines


def generate_srt(segments, first_speech_time: float = 0.0, max_chars_per_line: int = 40):
    """
    Generate SRT text, split into sentences, each block max 2 lines.
    """
    lines = []
    first_adjusted = False
    index = 1

    for seg in segments:
        start, end = seg.start, seg.end
        if not first_adjusted and first_speech_time > 0:
            start = first_speech_time
            first_adjusted = True

        # Split text into sentences
        sentences = split_sentences(seg.text.strip())

        # Distribute timing across sentences
        total_chars = sum(len(s) for s in sentences)
        current_time = start

        for s in sentences:
            proportion = len(s) / total_chars if total_chars > 0 else 0
            duration = proportion * (end - start)
            s_start, s_end = current_time, current_time + duration
            current_time = s_end

            # Wrap into max 2 lines
            wrapped_lines = wrap_text_to_two_lines(s, max_chars_per_line)

            # Build SRT block
            lines.append(str(index))
            lines.append(f"{format_srt_time(s_start)} --> {format_srt_time(s_end)}")
            lines.extend(wrapped_lines)
            lines.append("")  # blank line

            index += 1

    return "\n".join(lines)

@app.post("/transcribe_srt")
async def transcribe_to_srt(video: UploadFile = File(...)):
    unique_filename = f"{uuid.uuid4()}_{video.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        whisper_model = get_model()
        first_speech_time = detect_first_speech(save_path)

        segments, info = whisper_model.transcribe(
            save_path,
            beam_size=5,
            initial_prompt=None,
            condition_on_previous_text=False,
        )

        srt_content = generate_srt(segments, first_speech_time)
        srt_filename = unique_filename.rsplit(".", 1)[0] + ".srt"
        srt_path = os.path.join(UPLOAD_DIR, srt_filename)

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        return FileResponse(srt_path, media_type="text/plain", filename=srt_filename)

    except Exception as e:
        print("Error during SRT transcription:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate SRT: {str(e)}")

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

# -----------------------
# SPA fallback (MUST BE LAST)
# -----------------------
@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    return FileResponse(os.path.join(build_path, "index.html"))

# -----------------------
# Debug exception handler
# -----------------------
@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc()
        },
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Uvicorn on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
