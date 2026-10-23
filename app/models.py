"""
Pydantic models matching the Vapi Custom TTS webhook contract.

Vapi sends:
  POST /vapi/tts
  {
    "message": {
      "type": "voice-request",
      "text": "Hello, how can I help you?",
      "sampleRate": 24000,
      "timestamp": 1677123456789,
      "call": { "id": "...", "orgId": "..." },       # optional
      "assistant": { "id": "...", "name": "..." },   # optional
      "customer": { "number": "..." }                # optional
    }
  }

Response: HTTP 200, Content-Type: application/octet-stream
Body: raw mono 16-bit signed little-endian PCM at exactly sampleRate Hz
"""

from typing import Optional, Literal, Union
from pydantic import BaseModel, field_validator, model_validator

# Sample rates documented by Vapi (and validated in their example code)
SUPPORTED_SAMPLE_RATES = {8000, 16000, 22050, 24000}

# Text limits
MIN_TEXT_LEN = 1
MAX_TEXT_LEN = 2000  # safety cap; adjust if needed


class CallInfo(BaseModel):
    id: Optional[str] = None
    orgId: Optional[str] = None


class AssistantInfo(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None


class CustomerInfo(BaseModel):
    number: Optional[str] = None


class VoiceInfo(BaseModel):
    voiceId: Optional[str] = None
    id: Optional[str] = None

    class Config:
        extra = "allow"


class VapiMessage(BaseModel):
    type: Literal["voice-request"]
    text: str
    sampleRate: int
    timestamp: Optional[int] = None
    voiceId: Optional[str] = None
    voice: Optional[Union[VoiceInfo, str]] = None
    call: Optional[CallInfo] = None
    assistant: Optional[AssistantInfo] = None
    customer: Optional[CustomerInfo] = None

    class Config:
        extra = "allow"

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < MIN_TEXT_LEN:
            raise ValueError("text must not be empty or whitespace-only")
        if len(stripped) > MAX_TEXT_LEN:
            raise ValueError(
                f"text exceeds maximum length of {MAX_TEXT_LEN} characters"
            )
        return stripped

    @field_validator("sampleRate")
    @classmethod
    def sample_rate_supported(cls, v: int) -> int:
        if v not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sampleRate {v} is not supported. "
                f"Supported rates: {sorted(SUPPORTED_SAMPLE_RATES)}"
            )
        return v


class VapiTTSRequest(BaseModel):
    message: VapiMessage


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    voice_profile: Optional[str] = None
    warmup_complete: bool
