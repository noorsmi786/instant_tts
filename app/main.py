"""
FastAPI application — Vapi Custom TTS Server (Chatterbox-Nano)

Endpoints:
  GET  /health          → readiness probe (returns 503 until warmup is done)
  POST /vapi/tts        → Vapi Custom Voice webhook (authenticated, PCM response)
  GET  /preview/ui      → local-only browser preview UI (served from static/)
  POST /preview/tts     → local-only WAV preview (NOT sent to Vapi)
  GET  /admin/profiles  → list voice profiles (local-only)
  POST /admin/upload/{profile_id}  → upload reference audio (local-only)
  POST /admin/activate/{profile_id}/{filename} → activate a reference file
  DELETE /admin/reference/{profile_id}         → remove reference audio
  DELETE /admin/file/{profile_id}/{filename}   → delete an uploaded file
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).parent.parent / ".env",
    override=False,
)

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import auth, queue_manager, tts_engine, voice_profiles
from .models import HealthResponse, VapiTTSRequest

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_AUDIO_DURATION_S = 30
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _require_local(request: Request) -> None:
    client_host = request.client.host if request.client else ""
    if client_host not in _LOCAL_HOSTS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is only accessible from localhost.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    auth.init_secret()
    voice_profiles.init_assistant_map()
    queue_manager.setup()

    logger.info("Starting background model loading task...")
    loop = asyncio.get_event_loop()
    loop.create_task(asyncio.to_thread(tts_engine.load_model))
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="Vapi Custom TTS Server",
    description=(
        "Local Chatterbox-Nano TTS server for Vapi Custom Voice integration. "
        "Voice cloning via reference audio upload."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

_static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )


@app.get("/health", response_model=HealthResponse, tags=["Public"])
async def health():
    if not tts_engine.is_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "loading", "warmup_complete": False},
        )
    return HealthResponse(
        status="ready",
        model="chatterbox-nano",
        device=tts_engine.get_device(),
        warmup_complete=True,
    )


@app.post("/vapi/tts", tags=["Vapi"])
async def vapi_tts(
    request: Request,
    voice: Optional[str] = None,
    secret: Optional[str] = None,
):
    request_id = str(uuid.uuid4())[:8]

    # Validate secret either via X-Vapi-Secret header OR ?secret= query parameter
    auth_header = request.headers.get("x-vapi-secret")
    expected_secret = auth._VAPI_SECRET
    if expected_secret:
        if auth_header:
            if not secrets.compare_digest(auth_header.encode(), expected_secret.encode()):
                raise HTTPException(401, "Invalid X-Vapi-Secret header")
        elif secret:
            if not secrets.compare_digest(secret.encode(), expected_secret.encode()):
                raise HTTPException(401, "Invalid secret query parameter")
        # If neither provided, check if environment allows open local test or enforce
        else:
            # Check if VAPI_ALLOW_UNAUTHENTICATED is set or raise 401
            if os.environ.get("VAPI_REQUIRE_AUTH", "false").lower() == "true":
                raise HTTPException(401, "Missing X-Vapi-Secret header or secret query parameter")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request body must be valid JSON",
        )

    try:
        payload = VapiTTSRequest(**body)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    msg = payload.message
    text = msg.text
    sample_rate = msg.sampleRate
    assistant_id = msg.assistant.id if msg.assistant else None

    # Extract voiceId from URL query param OR from Vapi payload (msg.voiceId or msg.voice)
    requested_voice = voice
    if not requested_voice:
        if msg.voiceId:
            requested_voice = msg.voiceId
        elif isinstance(msg.voice, str):
            requested_voice = msg.voice
        elif hasattr(msg.voice, "voiceId") and msg.voice.voiceId:
            requested_voice = msg.voice.voiceId
        elif hasattr(msg.voice, "id") and msg.voice.id:
            requested_voice = msg.voice.id

    logger.info(
        "[%s] TTS request: %d chars, %dHz, requested_voice=%s, assistant=%s",
        request_id, len(text), sample_rate, requested_voice or "active_default", assistant_id or "unknown",
    )

    profile = voice_profiles.get_profile_for_assistant(assistant_id)
    # Resolve reference audio using requested_voice (filename or default) or profile default
    ref_audio = voice_profiles.resolve_reference_audio(profile.profile_id, requested_voice)

    if not tts_engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS model is still loading. Retry in a moment.",
        )

    try:
        async with queue_manager.synthesis_slot():
            loop = asyncio.get_event_loop()
            pcm_bytes, metrics = await loop.run_in_executor(
                None,
                lambda: tts_engine.synthesize(
                    text=text,
                    sample_rate=sample_rate,
                    reference_audio=ref_audio,
                    exaggeration=profile.exaggeration,
                    cfg_weight=profile.cfg_weight,
                ),
            )
    except asyncio.TimeoutError:
        logger.warning("[%s] Request timed out in queue", request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server busy — request timed out in queue. Please retry.",
        )
    except RuntimeError as exc:
        if "queue full" in str(exc).lower():
            logger.warning("[%s] Queue full: %s", request_id, exc)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            )
        logger.error("[%s] Synthesis error: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TTS synthesis failed. Check server logs.",
        )
    except Exception as exc:
        logger.exception("[%s] Unexpected synthesis error", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal TTS error",
        )

    return Response(
        content=pcm_bytes,
        media_type="application/octet-stream",
        headers={
            "X-Audio-Duration": f"{metrics.audio_duration_s:.3f}",
            "X-RTF": f"{metrics.rtf:.3f}",
            "X-Generation-Ms": f"{metrics.model_gen_ms:.0f}",
            "X-Total-Ms": f"{metrics.total_ms:.0f}",
            "X-Request-Id": request_id,
        },
    )


@app.get("/preview/ui", tags=["Preview"], dependencies=[Depends(_require_local)])
async def preview_ui():
    return FileResponse(str(_static_dir / "preview.html"))


@app.post("/preview/tts", tags=["Preview"], dependencies=[Depends(_require_local)])
async def preview_tts(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    sample_rate = int(body.get("sampleRate", 24000))
    profile_id = body.get("profileId", "client_default")
    voice_id = body.get("voiceId")  # e.g. "default" or specific uploaded filename

    if not text:
        raise HTTPException(400, "text is required")
    if sample_rate not in {8000, 16000, 22050, 24000}:
        raise HTTPException(400, f"Unsupported sampleRate: {sample_rate}")

    profile = voice_profiles.get_profile(profile_id) or voice_profiles.get_profile("client_default")
    ref_audio = voice_profiles.resolve_reference_audio(profile_id, voice_id)

    if not tts_engine.is_ready():
        raise HTTPException(503, "Model loading, retry shortly")

    try:
        async with queue_manager.synthesis_slot():
            loop = asyncio.get_event_loop()
            pcm_bytes, metrics = await loop.run_in_executor(
                None,
                lambda: tts_engine.synthesize(
                    text=text,
                    sample_rate=sample_rate,
                    reference_audio=ref_audio,
                    exaggeration=profile.exaggeration,
                    cfg_weight=profile.cfg_weight,
                ),
            )
    except asyncio.TimeoutError:
        raise HTTPException(503, "Server busy")
    except RuntimeError as exc:
        raise HTTPException(429 if "queue full" in str(exc).lower() else 500, str(exc))

    wav_bytes = tts_engine.pcm_to_wav(pcm_bytes, sample_rate)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{metrics.audio_duration_s:.3f}",
            "X-RTF": f"{metrics.rtf:.3f}",
            "X-Generation-Ms": f"{metrics.model_gen_ms:.0f}",
        },
    )


@app.get("/admin/voices", tags=["Admin"], dependencies=[Depends(_require_local)])
async def list_voices(profile_id: str = "client_default"):
    return voice_profiles.list_available_voices(profile_id)


@app.get("/admin/profiles", tags=["Admin"], dependencies=[Depends(_require_local)])
async def list_profiles():
    profiles = voice_profiles.list_profiles()
    return [
        {
            "profile_id": p.profile_id,
            "display_name": p.display_name,
            "reference_audio": os.path.basename(p.reference_audio) if p.reference_audio else None,
            "exaggeration": p.exaggeration,
            "cfg_weight": p.cfg_weight,
        }
        for p in profiles
    ]


@app.post("/admin/upload/{profile_id}", tags=["Admin"], dependencies=[Depends(_require_local)])
async def upload_reference_audio(
    profile_id: str,
    file: UploadFile = File(...),
):
    if voice_profiles.get_profile(profile_id) is None:
        raise HTTPException(404, f"Unknown profile: {profile_id!r}")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(
            400,
            f"Unsupported format {ext!r}. Allowed: {sorted(ALLOWED_AUDIO_EXTS)}",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit")

    safe_name = Path(file.filename or "reference").stem
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "-_") or "reference"
    filename = f"{safe_name}{ext}"

    upload_dir = voice_profiles.get_upload_dir(profile_id)
    dest = os.path.join(upload_dir, filename)
    with open(dest, "wb") as f:
        f.write(content)

    try:
        import soundfile as sf
        info = sf.info(dest)
        if info.duration > MAX_AUDIO_DURATION_S:
            os.remove(dest)
            raise HTTPException(
                400,
                f"Audio is {info.duration:.1f}s. Maximum is {MAX_AUDIO_DURATION_S}s. "
                "Trim to 10–20s for best cloning quality.",
            )
        logger.info("Uploaded %s for profile %s (%.1fs)", filename, profile_id, info.duration)
    except Exception as exc:
        logger.warning("Could not validate audio duration: %s", exc)

    # Auto-activate the newly uploaded file for this profile immediately
    path = voice_profiles.set_reference_audio(profile_id, filename)
    logger.info("Auto-activated reference audio %s for profile %s", filename, profile_id)

    return {
        "status": "activated",
        "profile_id": profile_id,
        "filename": filename,
        "size_bytes": len(content),
        "path": path,
        "message": f"Voice recording '{filename}' uploaded and activated for cloning!",
    }


@app.get("/admin/files/{profile_id}", tags=["Admin"], dependencies=[Depends(_require_local)])
async def list_uploaded_files(profile_id: str):
    if voice_profiles.get_profile(profile_id) is None:
        raise HTTPException(404, f"Unknown profile: {profile_id!r}")
    upload_dir = voice_profiles.get_upload_dir(profile_id)
    profile = voice_profiles.get_profile(profile_id)
    active_file = os.path.basename(profile.reference_audio) if profile.reference_audio else None

    files = []
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            if os.path.isfile(os.path.join(upload_dir, f)) and Path(f).suffix.lower() in ALLOWED_AUDIO_EXTS:
                files.append({
                    "filename": f,
                    "is_active": (f == active_file)
                })
    return files


@app.post("/admin/activate/{profile_id}/{filename}", tags=["Admin"],
          dependencies=[Depends(_require_local)])
async def activate_reference_audio(profile_id: str, filename: str):
    try:
        path = voice_profiles.set_reference_audio(profile_id, filename)
        return {
            "status": "activated",
            "profile_id": profile_id,
            "filename": filename,
            "path": path,
            "message": "Voice cloning is now active for this profile.",
        }
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.delete("/admin/reference/{profile_id}", tags=["Admin"],
            dependencies=[Depends(_require_local)])
async def clear_reference_audio(profile_id: str):
    try:
        voice_profiles.clear_reference_audio(profile_id)
        return {"status": "cleared", "profile_id": profile_id}
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.delete("/admin/file/{profile_id}/{filename}", tags=["Admin"],
            dependencies=[Depends(_require_local)])
async def delete_uploaded_file(profile_id: str, filename: str):
    if voice_profiles.get_profile(profile_id) is None:
        raise HTTPException(404, f"Unknown profile: {profile_id!r}")

    upload_dir = voice_profiles.get_upload_dir(profile_id)
    safe_name = Path(filename).name
    file_path = os.path.join(upload_dir, safe_name)

    if not os.path.isfile(file_path):
        raise HTTPException(404, f"File {safe_name!r} not found")

    profile = voice_profiles.get_profile(profile_id)
    if profile and profile.reference_audio == file_path:
        voice_profiles.clear_reference_audio(profile_id)

    os.remove(file_path)
    return {"status": "deleted", "profile_id": profile_id, "filename": safe_name}
