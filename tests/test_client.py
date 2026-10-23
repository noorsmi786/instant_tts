"""
Local test client — calls the running TTS server, saves raw PCM,
and wraps it in WAV for listening.

Usage:
  python tests/test_client.py
"""

import argparse
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

SERVER_URL = os.environ.get("TTS_SERVER_URL", "http://127.0.0.1:8000")
VAPI_SECRET = os.environ.get("VAPI_SECRET", "")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    n_ch = 1
    bits = 16
    data_len = len(pcm_bytes)
    buf = bytearray()
    buf += b"RIFF"
    buf += struct.pack("<I", 36 + data_len)
    buf += b"WAVE"
    buf += b"fmt "
    buf += struct.pack("<I", 16)
    buf += struct.pack("<H", 1)
    buf += struct.pack("<H", n_ch)
    buf += struct.pack("<I", sample_rate)
    buf += struct.pack("<I", sample_rate * n_ch * bits // 8)
    buf += struct.pack("<H", n_ch * bits // 8)
    buf += struct.pack("<H", bits)
    buf += b"data"
    buf += struct.pack("<I", data_len)
    buf += pcm_bytes
    return bytes(buf)


def check_health():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        if r.status_code == 200 and r.json().get("status") == "ready":
            print(f"[OK] Server is ready: {r.json()}")
            return True
        print(f"[WAIT] Server not ready yet: {r.json()}")
        return False
    except requests.ConnectionError:
        print(f"[ERROR] Cannot connect to {SERVER_URL}. Is the server running?")
        return False


def run_test(text: str, sample_rate: int) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not VAPI_SECRET:
        print("[ERROR] VAPI_SECRET is not set.")
        sys.exit(1)

    payload = {
        "message": {
            "type": "voice-request",
            "text": text,
            "sampleRate": sample_rate,
        }
    }

    headers = {
        "Content-Type": "application/json",
        "x-vapi-secret": VAPI_SECRET,
    }

    print(f"\nSending TTS request:")
    print(f"   Text ({len(text)} chars): {text[:80]}")
    print(f"   Sample rate: {sample_rate} Hz")

    t0 = time.monotonic()
    try:
        r = requests.post(
            f"{SERVER_URL}/vapi/tts",
            json=payload,
            headers=headers,
            timeout=120,
        )
    except requests.Timeout:
        print("[ERROR] Request timed out (>120s)")
        sys.exit(1)
    except requests.ConnectionError as e:
        print(f"[ERROR] Connection error: {e}")
        sys.exit(1)

    elapsed_ms = (time.monotonic() - t0) * 1000

    if r.status_code != 200:
        print(f"[ERROR] Server returned {r.status_code}: {r.text}")
        sys.exit(1)

    pcm_bytes = r.content
    content_type = r.headers.get("content-type", "")

    assert "application/octet-stream" in content_type, f"Wrong content-type: {content_type}"
    assert len(pcm_bytes) > 0, "Empty response body"
    assert len(pcm_bytes) % 2 == 0, "PCM byte count is odd — not valid 16-bit"
    assert pcm_bytes[:4] != b"RIFF", "Response contains WAV header (should be raw PCM)"

    audio_dur = float(r.headers.get("x-audio-duration", 0))
    rtf = float(r.headers.get("x-rtf", 0))
    gen_ms = float(r.headers.get("x-generation-ms", 0))
    total_ms = float(r.headers.get("x-total-ms", elapsed_ms))

    print(f"\n[SUCCESS]")
    print(f"   PCM bytes:        {len(pcm_bytes):,}")
    print(f"   Audio duration:   {audio_dur:.2f}s")
    print(f"   Model gen time:   {gen_ms:.0f}ms")
    print(f"   Total latency:    {total_ms:.0f}ms (roundtrip: {elapsed_ms:.0f}ms)")
    print(f"   Real-time factor: {rtf:.3f}")

    ts = int(time.time())
    pcm_path = os.path.join(OUTPUT_DIR, f"output_{ts}_{sample_rate}hz.pcm")
    wav_path = os.path.join(OUTPUT_DIR, f"output_{ts}_{sample_rate}hz.wav")

    with open(pcm_path, "wb") as f:
        f.write(pcm_bytes)
    with open(wav_path, "wb") as f:
        f.write(pcm_to_wav(pcm_bytes, sample_rate))

    print(f"\n   Raw PCM saved: {pcm_path}")
    print(f"   WAV saved:     {wav_path}")


def main():
    parser = argparse.ArgumentParser(description="Local TTS endpoint test client")
    parser.add_argument(
        "--text",
        default="Hello! This is a test of the Chatterbox voice cloning server connected to Vapi.",
        help="Text to synthesise",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=24000,
        choices=[8000, 16000, 22050, 24000],
        help="Sample rate in Hz",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        load_dotenv(env_path)
        global VAPI_SECRET
        VAPI_SECRET = os.environ.get("VAPI_SECRET", "")
    except ImportError:
        pass

    print(f"Server URL: {SERVER_URL}")
    if not check_health():
        sys.exit(1)

    run_test(args.text, args.rate)


if __name__ == "__main__":
    main()
