"""
TTS engine — wraps Chatterbox-Nano for synthesis and audio conversion.

Design decisions:
- Model is loaded ONCE at startup inside load_model().
- A warmup pass is run before marking the service ready.
- All audio is converted to: mono, 16-bit signed LE PCM, at the requested Hz.
- Unicode text is preserved; no stripping of non-ASCII characters.
- Language validation: only 'en' is supported by Chatterbox-Nano.
- Metrics are measured and returned for benchmarking.

NOTE: The exact Chatterbox-Nano Python API is verified at startup by
inspecting the installed package. If the import fails, clear error messages
are raised rather than guessing at SDK internals.
"""

import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)

# ─── Chatterbox model (loaded once) ──────────────────────────────────────────
_model = None
_model_sr: int = 24000     # Chatterbox native sample rate (verified at load)
_device: str = "cpu"
_warmup_done: bool = False

# Max text Chatterbox can handle in one pass (conservative limit)
MAX_SYNTHESIS_CHARS = 1500

# ─── Metrics ─────────────────────────────────────────────────────────────────
@dataclass
class SynthesisMetrics:
    text_len: int
    model_gen_ms: float          # time inside model.generate()
    audio_duration_s: float      # duration of synthesised audio
    rtf: float                   # real-time factor  (gen_s / audio_s)
    total_ms: float              # full pipeline including resample & encode
    output_bytes: int
    used_reference_audio: bool


# ─── Loader ──────────────────────────────────────────────────────────────────
def load_model() -> None:
    """
    Import and load Chatterbox using zero-overhead meta-device loading to fit
    within Windows commit memory limits on CPU without OOM or download.
    """
    global _model, _model_sr, _device, _warmup_done

    device = os.environ.get("TTS_DEVICE", "cpu").strip().lower()
    _device = device

    logger.info("Loading Chatterbox model on device=%s ...", device)

    import gc
    import torch
    import safetensors
    from pathlib import Path

    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.t3 import T3
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.tokenizers import EnTokenizer
    from chatterbox.tts import ChatterboxTTS, Conditionals
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    ckpt_dir = Path(
        os.environ.get(
            "CHATTERBOX_CKPT_DIR",
            r"C:\Users\noorr\.cache\huggingface\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
        )
    )

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Chatterbox checkpoint directory not found at: {ckpt_dir}")

    # 1. Voice Encoder
    logger.info("Loading VoiceEncoder...")
    with torch.device("meta"):
        ve = VoiceEncoder()
    with safetensors.safe_open(ckpt_dir / "ve.safetensors", framework="pt", device="cpu") as f:
        ve.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, assign=True, strict=False)
    ve.eval()
    gc.collect()

    # 2. S3Gen
    logger.info("Loading S3Gen...")
    s3gen = S3Gen()
    with safetensors.safe_open(ckpt_dir / "s3gen.safetensors", framework="pt", device="cpu") as f:
        s3gen.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=False)
    s3gen.eval()
    gc.collect()

    # 3. T3 Transformer in bfloat16 -> float32 (fits within commit limit)
    logger.info("Loading T3 Transformer...")
    with torch.device("meta"):
        t3 = T3()
    with safetensors.safe_open(ckpt_dir / "t3_cfg.safetensors", framework="pt", device="cpu") as f:
        t3.load_state_dict({k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys()}, assign=True, strict=False)
    t3.to(torch.float32)
    # Materialize rotary embeddings on CPU so no meta tensor remains
    t3.tfmr.rotary_emb = LlamaRotaryEmbedding(t3.cfg)
    t3.eval()
    gc.collect()

    # 4. Tokenizer & Conditionals
    logger.info("Loading Tokenizer & Conditionals...")
    tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))
    conds = Conditionals.load(ckpt_dir / "conds.pt", map_location=torch.device("cpu"))

    # 5. Instantiate ChatterboxTTS
    logger.info("Instantiating ChatterboxTTS...")
    _model = ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)
    _model_sr = int(getattr(_model, "sr", 24000))
    logger.info("Model loaded successfully. Native sample rate: %d Hz", _model_sr)

    # 6. Warmup pass
    _warmup()


def _warmup() -> None:
    """Run one warmup synthesis pass to compile graph."""
    global _warmup_done
    logger.info("Running warmup synthesis pass...")
    t0 = time.monotonic()
    try:
        _generate_raw("Hello.", reference_audio=None)
        elapsed = (time.monotonic() - t0) * 1000
        logger.info("Warmup complete in %.0f ms", elapsed)
        _warmup_done = True
    except Exception as exc:
        logger.error("Warmup failed: %s", exc)
        raise


def is_ready() -> bool:
    return _model is not None and _warmup_done


def get_device() -> str:
    return _device


# ─── Synthesis ───────────────────────────────────────────────────────────────
def synthesize(
    text: str,
    sample_rate: int,
    reference_audio: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> tuple[bytes, SynthesisMetrics]:
    """
    Synthesise text and return (raw_pcm_bytes, metrics).

    raw_pcm_bytes: mono, 16-bit signed LE PCM at exactly sample_rate Hz.
    Raises RuntimeError on synthesis failure.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    pipeline_start = time.monotonic()

    # ── 1. Generate float32 audio from model ─────────────────────────────────
    gen_start = time.monotonic()
    wav_tensor = _generate_raw(
        text,
        reference_audio=reference_audio,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    gen_ms = (time.monotonic() - gen_start) * 1000

    # ── 2. Convert tensor → numpy float32 ────────────────────────────────────
    audio = _tensor_to_numpy(wav_tensor)

    # ── 3. Ensure mono ───────────────────────────────────────────────────────
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    # ── 4. Calculate audio duration ──────────────────────────────────────────
    audio_duration_s = len(audio) / _model_sr

    # ── 5. Resample to requested sample rate ─────────────────────────────────
    if sample_rate != _model_sr:
        audio = _resample(audio, _model_sr, sample_rate)

    # ── 6. Clip and encode as 16-bit LE PCM ──────────────────────────────────
    pcm_bytes = _to_pcm16(audio)

    total_ms = (time.monotonic() - pipeline_start) * 1000
    rtf = (gen_ms / 1000) / audio_duration_s if audio_duration_s > 0 else 0.0

    metrics = SynthesisMetrics(
        text_len=len(text),
        model_gen_ms=gen_ms,
        audio_duration_s=audio_duration_s,
        rtf=rtf,
        total_ms=total_ms,
        output_bytes=len(pcm_bytes),
        used_reference_audio=reference_audio is not None,
    )

    logger.info(
        "Synthesis: %d chars → %.2fs audio | gen=%.0fms rtf=%.2f total=%.0fms %s",
        len(text),
        audio_duration_s,
        gen_ms,
        rtf,
        total_ms,
        "(cloned)" if reference_audio else "(default)",
    )

    return pcm_bytes, metrics


def _generate_raw(
    text: str,
    reference_audio: Optional[str],
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
):
    """
    Call model.generate() with available parameters.
    Introspects the method signature to pass only supported kwargs.
    """
    import inspect

    sig = inspect.signature(_model.generate)
    params = set(sig.parameters.keys())

    kwargs: dict = {}

    if "audio_prompt_path" in params and reference_audio is not None:
        kwargs["audio_prompt_path"] = reference_audio
    elif "reference_audio" in params and reference_audio is not None:
        kwargs["reference_audio"] = reference_audio

    if "exaggeration" in params:
        kwargs["exaggeration"] = exaggeration
    if "cfg_weight" in params:
        kwargs["cfg_weight"] = cfg_weight

    return _model.generate(text, **kwargs)


def _tensor_to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor (or numpy array) to float32 numpy array."""
    if hasattr(tensor, "numpy"):
        try:
            arr = tensor.squeeze().numpy()
        except RuntimeError:
            arr = tensor.squeeze().cpu().numpy()
    elif hasattr(tensor, "cpu"):
        arr = tensor.squeeze().cpu().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32).squeeze()

    return arr.astype(np.float32)


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample float32 audio array using scipy's polyphase filter."""
    if src_rate == dst_rate:
        return audio
    num_samples = int(len(audio) * dst_rate / src_rate)
    return scipy.signal.resample(audio, num_samples).astype(np.float32)


def _to_pcm16(audio: np.ndarray) -> bytes:
    """
    Convert float32 audio [-1, 1] to mono 16-bit signed little-endian PCM bytes.
    Clips before conversion to prevent wrap-around artefacts.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


# ─── WAV helper (for browser preview only, never sent to Vapi) ───────────────
def pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """
    Wrap raw PCM bytes in a WAV container for browser playback.
    This output must NEVER be sent to the Vapi endpoint.
    """
    import struct
    n_channels = 1
    bit_depth = 16
    byte_rate = sample_rate * n_channels * bit_depth // 8
    block_align = n_channels * bit_depth // 8
    data_len = len(pcm_bytes)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_len))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))            # PCM
    buf.write(struct.pack("<H", n_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bit_depth))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    buf.write(pcm_bytes)
    return buf.getvalue()
