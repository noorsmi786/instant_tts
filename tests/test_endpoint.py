"""
Test suite for the Vapi Custom TTS server.

Tests:
  - Valid Vapi requests
  - Missing / incorrect authentication
  - Missing message or text
  - Incorrect message type
  - Unsupported sample rates
  - Empty or excessive text
  - PCM byte format verification
  - Correct response headers

Run with:
  pytest tests/test_endpoint.py -v
"""

import os
import struct
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Set up environment before importing app
os.environ["VAPI_SECRET"] = "vapi_sec_test12345"
os.environ["TTS_MAX_CONCURRENT"] = "1"
os.environ["TTS_QUEUE_SIZE"] = "4"
os.environ["TTS_REQUEST_TIMEOUT"] = "30"

from app.models import HealthResponse, VapiTTSRequest
from app.tts_engine import SynthesisMetrics

VALID_SECRET = "vapi_sec_test12345"
WRONG_SECRET = "wrong-secret"
HEADERS_OK = {"x-vapi-secret": VALID_SECRET}

VALID_BODY = {
    "message": {
        "type": "voice-request",
        "text": "Hello, how can I help you today?",
        "sampleRate": 24000,
    }
}

_SILENCE_PCM = bytes(24000 * 2)

_MOCK_METRICS = SynthesisMetrics(
    text_len=30,
    model_gen_ms=200,
    audio_duration_s=1.0,
    rtf=0.2,
    total_ms=250,
    output_bytes=len(_SILENCE_PCM),
    used_reference_audio=False,
)


@pytest.fixture
def client():
    with patch("app.tts_engine.load_model", return_value=None), \
         patch("app.tts_engine.is_ready", return_value=True), \
         patch("app.tts_engine.synthesize", return_value=(_SILENCE_PCM, _MOCK_METRICS)):
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestAuthentication:
    def test_missing_secret_returns_401(self, client):
        r = client.post("/vapi/tts", json=VALID_BODY, headers={})
        assert r.status_code == 401

    def test_wrong_secret_returns_401(self, client):
        r = client.post("/vapi/tts", json=VALID_BODY, headers={"x-vapi-secret": WRONG_SECRET})
        assert r.status_code == 401

    def test_correct_secret_returns_200(self, client):
        with patch("app.tts_engine.is_ready", return_value=True), \
             patch("app.tts_engine.synthesize", return_value=(_SILENCE_PCM, _MOCK_METRICS)):
            r = client.post("/vapi/tts", json=VALID_BODY, headers=HEADERS_OK)
        assert r.status_code == 200


class TestRequestValidation:
    def test_missing_message_returns_400(self, client):
        r = client.post("/vapi/tts", json={}, headers=HEADERS_OK)
        assert r.status_code == 400

    def test_missing_text_returns_400(self, client):
        body = {"message": {"type": "voice-request", "sampleRate": 24000}}
        r = client.post("/vapi/tts", json=body, headers=HEADERS_OK)
        assert r.status_code == 400

    def test_wrong_message_type_returns_400(self, client):
        body = {"message": {"type": "invalid-type", "text": "Hello", "sampleRate": 24000}}
        r = client.post("/vapi/tts", json=body, headers=HEADERS_OK)
        assert r.status_code == 400

    def test_empty_text_returns_400(self, client):
        body = {"message": {"type": "voice-request", "text": "   ", "sampleRate": 24000}}
        r = client.post("/vapi/tts", json=body, headers=HEADERS_OK)
        assert r.status_code == 400

    def test_unsupported_sample_rate_returns_400(self, client):
        for bad_rate in [11025, 32000, 48000, 96000]:
            body = {"message": {"type": "voice-request", "text": "Hello", "sampleRate": bad_rate}}
            r = client.post("/vapi/tts", json=body, headers=HEADERS_OK)
            assert r.status_code == 400

    def test_valid_sample_rates_accepted(self, client):
        for rate in [8000, 16000, 22050, 24000]:
            body = {"message": {"type": "voice-request", "text": "Hello", "sampleRate": rate}}
            with patch("app.tts_engine.is_ready", return_value=True), \
                 patch("app.tts_engine.synthesize", return_value=(_SILENCE_PCM, _MOCK_METRICS)):
                r = client.post("/vapi/tts", json=body, headers=HEADERS_OK)
            assert r.status_code == 200

    def test_unicode_text_preserved(self, client):
        body = {
            "message": {
                "type": "voice-request",
                "text": "Héllo wörld — こんにちは",
                "sampleRate": 24000,
            }
        }
        with patch("app.tts_engine.is_ready", return_value=True), \
             patch("app.tts_engine.synthesize", return_value=(_SILENCE_PCM, _MOCK_METRICS)):
            r = client.post("/vapi/tts", json=body, headers=HEADERS_OK)
        assert r.status_code == 200


class TestResponseFormat:
    def test_content_type_is_octet_stream(self, client):
        with patch("app.tts_engine.is_ready", return_value=True), \
             patch("app.tts_engine.synthesize", return_value=(_SILENCE_PCM, _MOCK_METRICS)):
            r = client.post("/vapi/tts", json=VALID_BODY, headers=HEADERS_OK)
        assert r.status_code == 200
        assert "application/octet-stream" in r.headers["content-type"]

    def test_pcm_byte_count_is_even(self, client):
        with patch("app.tts_engine.is_ready", return_value=True), \
             patch("app.tts_engine.synthesize", return_value=(_SILENCE_PCM, _MOCK_METRICS)):
            r = client.post("/vapi/tts", json=VALID_BODY, headers=HEADERS_OK)
        assert r.status_code == 200
        assert len(r.content) % 2 == 0

    def test_no_wav_header_in_vapi_response(self, client):
        """Must return RAW PCM to Vapi (NO 'RIFF' header)."""
        with patch("app.tts_engine.is_ready", return_value=True), \
             patch("app.tts_engine.synthesize", return_value=(_SILENCE_PCM, _MOCK_METRICS)):
            r = client.post("/vapi/tts", json=VALID_BODY, headers=HEADERS_OK)
        assert r.status_code == 200
        assert r.content[:4] != b"RIFF"


class TestHealth:
    def test_health_ready(self, client):
        with patch("app.tts_engine.is_ready", return_value=True):
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"
