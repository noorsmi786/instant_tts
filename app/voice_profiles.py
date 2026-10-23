"""
Voice profile registry.

Maps authenticated client/assistant identities to approved voice profiles.
No arbitrary filesystem paths or unvalidated voice IDs are accepted.

Each profile specifies:
- profile_id:       unique slug (no path separators)
- display_name:     human-readable label
- reference_audio:  absolute path to uploaded WAV (None = use default Nano voice)
- exaggeration:     emotion exaggeration scalar (0.0–1.0, Chatterbox param)
- cfg_weight:       classifier-free guidance weight (0.0–1.0)
- language:         ISO code (en only for Nano MVP)

Add more profiles below as needed; keep client voice data separate.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# Base upload directory — profiles may only reference files inside here
_BASE_UPLOAD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "uploads")
)


@dataclass
class VoiceProfile:
    profile_id: str          # e.g. "client_default"
    display_name: str
    reference_audio: Optional[str]  # absolute path inside uploads/, or None
    exaggeration: float = 0.5       # Chatterbox emotion exaggeration
    cfg_weight: float = 0.5         # Chatterbox CFG weight
    language: str = "en"


# ─── Registry ─────────────────────────────────────────────────────────────────
# Add one entry per client. For MVP, one client, one profile.

_REGISTRY: dict[str, VoiceProfile] = {
    "client_default": VoiceProfile(
        profile_id="client_default",
        display_name="Default Client Voice",
        reference_audio=None,   # None = Chatterbox Nano default voice (no cloning)
    ),
}

# Map Vapi assistant IDs → profile IDs (populated from .env at startup)
# e.g.  VAPI_ASSISTANT_ID_CLIENT_DEFAULT=asst_abc123
_ASSISTANT_TO_PROFILE: dict[str, str] = {}


def init_assistant_map() -> None:
    """
    Read VAPI_ASSISTANT_ID_* env vars and build assistant→profile mapping.
    Called from main.py after dotenv is loaded.
    """
    for key, value in os.environ.items():
        if key.startswith("VAPI_ASSISTANT_ID_"):
            profile_id = key[len("VAPI_ASSISTANT_ID_"):].lower()
            if profile_id in _REGISTRY:
                _ASSISTANT_TO_PROFILE[value.strip()] = profile_id


def get_profile_for_assistant(assistant_id: Optional[str]) -> VoiceProfile:
    """Return the voice profile for a given Vapi assistant ID, or the default."""
    if assistant_id and assistant_id in _ASSISTANT_TO_PROFILE:
        profile_id = _ASSISTANT_TO_PROFILE[assistant_id]
        return _REGISTRY[profile_id]
    return _REGISTRY["client_default"]


def get_profile(profile_id: str) -> Optional[VoiceProfile]:
    """Return a profile by ID, or None if not found."""
    return _REGISTRY.get(profile_id)


def list_profiles() -> list[VoiceProfile]:
    return list(_REGISTRY.values())


def list_available_voices(profile_id: str = "client_default") -> list[dict]:
    """List all available voices: default model voice + all uploaded audio files."""
    voices = [
        {
            "id": "default",
            "name": "Default Model Voice (Standard)",
            "filename": None,
            "is_active": (_REGISTRY[profile_id].reference_audio is None) if profile_id in _REGISTRY else True
        }
    ]
    upload_dir = get_upload_dir(profile_id)
    active_path = _REGISTRY[profile_id].reference_audio if profile_id in _REGISTRY else None
    active_filename = os.path.basename(active_path) if active_path else None

    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
                voices.append({
                    "id": f,
                    "name": f"Custom Voice: {f}",
                    "filename": f,
                    "is_active": (f == active_filename)
                })
    return voices


def resolve_reference_audio(profile_id: str, voice_id: Optional[str] = None) -> Optional[str]:
    """
    Resolve the absolute audio path for a given voice_id or the profile's active voice.
    If voice_id is 'default' or None and profile has no active reference, returns None.
    """
    if voice_id and voice_id != "default":
        # Check if voice_id is an uploaded file
        upload_dir = get_upload_dir(profile_id)
        candidate = os.path.abspath(os.path.join(upload_dir, voice_id))
        if candidate.startswith(_BASE_UPLOAD_DIR + os.sep) and os.path.isfile(candidate):
            return candidate

    profile = get_profile(profile_id)
    return profile.reference_audio if profile else None


def set_reference_audio(profile_id: str, filename: Optional[str]) -> Optional[str]:
    """
    Set the reference audio for a profile.
    If filename is None or 'default', clears reference audio (reverts to default).
    Returns the absolute path or None.
    """
    if profile_id not in _REGISTRY:
        raise ValueError(f"Unknown profile: {profile_id!r}")

    if not filename or filename == "default":
        _REGISTRY[profile_id].reference_audio = None
        return None

    # Build and validate the absolute path — no path traversal allowed
    candidate = os.path.abspath(
        os.path.join(_BASE_UPLOAD_DIR, profile_id, filename)
    )
    if not candidate.startswith(_BASE_UPLOAD_DIR + os.sep):
        raise ValueError("Invalid filename: path traversal detected")
    if not os.path.isfile(candidate):
        raise ValueError(f"File not found: {filename!r} for profile {profile_id!r}")

    _REGISTRY[profile_id].reference_audio = candidate
    return candidate


def clear_reference_audio(profile_id: str) -> None:
    """Remove the reference audio from a profile (revert to default voice)."""
    if profile_id not in _REGISTRY:
        raise ValueError(f"Unknown profile: {profile_id!r}")
    _REGISTRY[profile_id].reference_audio = None


def get_upload_dir(profile_id: str) -> str:
    """Return the upload directory for a profile (guaranteed safe path)."""
    safe_id = profile_id.replace("/", "").replace("..", "")
    path = os.path.join(_BASE_UPLOAD_DIR, safe_id)
    os.makedirs(path, exist_ok=True)
    return path
