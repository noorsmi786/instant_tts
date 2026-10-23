# Vapi Custom TTS & Zero-Shot Voice Cloning Server

> **Enterprise-Ready Self-Hosted Text-to-Speech Engine**  
> Replaces expensive paid voice APIs (such as ElevenLabs) in your existing **Vapi voice agents**.  
> Features **instant zero-shot voice cloning**, dynamic multi-voice selection via Vapi's `Voice ID`, and full compliance with Vapi's raw PCM audio webhook specifications.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Key Features](#key-features)
3. [Language Support & Model Roadmap](#language-support--model-roadmap)
4. [Hardware Requirements (CPU vs. GPU)](#hardware-requirements-cpu-vs-gpu)
5. [Local Development Quickstart (Windows/Linux)](#local-development-quickstart)
6. [Testing the Engine (Browser Preview & API)](#testing-the-engine)
7. [Vapi Dashboard Configuration](#vapi-dashboard-configuration)
8. [Production Deployment Guide (Linux VPS / GPU Server)](#production-deployment-guide)
9. [API Reference & Webhook Contract](#api-reference--webhook-contract)
10. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## 1. Architecture Overview

```
[ Customer Phone / Web Widget ]
              │
              ▼
       [ Vapi Cloud ]
    (Transcribe + LLM)
              │
              │  POST /vapi/tts (JSON Payload + Voice ID)
              ▼
   [ HTTPS Tunnel / Domain ]
   (ngrok / Cloudflare / Nginx)
              │
              ▼
┌────────────────────────────────────────────────────────┐
│  Custom FastAPI Voice Server (Port 8000)               │
│                                                        │
│  ├── Auth & Header Validation (X-Vapi-Secret)          │
│  ├── Concurrency Control (Bounded Queue Semaphore)     │
│  ├── Neural Voice Encoder (Extracts Speaker Vector)    │
│  ├── Transformer / Diffusion TTS Engine                │
│  └── Audio Pipeline (Float32 -> Mono 16-bit LE PCM)    │
└────────────────────────────────────────────────────────┘
              │
              │  200 OK: application/octet-stream (Raw PCM)
              ▼
       [ Vapi Cloud ]
              │
              ▼
[ Customer Hears Cloned Voice (<300ms on GPU) ]
```

---

## 2. Key Features

- **100% Vapi Webhook Compatibility**: Strictly delivers uncompressed, headerless, signed 16-bit little-endian mono PCM audio matching requested sample rates (8kHz, 16kHz, 22.05kHz, 24kHz).
- **Instant Zero-Shot Voice Cloning**: Upload a single 5–15 second audio sample (`.wav`, `.mp3`, `.m4a`). No model retraining or fine-tuning required.
- **Dynamic Multi-Voice Routing**: Switch voices dynamically per assistant or per call using Vapi's native `voiceId` field or URL query parameter (`?voice=my_voice.mp3`).
- **Memory-Efficient Architecture**: Custom zero-overhead streaming loader using PyTorch `meta` device to fit large neural models (~3.5 GB) into constrained virtual memory systems without out-of-memory errors.
- **Zero Ongoing API Fees**: Fully self-hosted on your own hardware or private cloud instance.

---

## 3. Language Support & Model Roadmap

| Model Engine | Supported Languages | Voice Cloning Capability | Best Use Case | License |
| :--- | :--- | :--- | :--- | :--- |
| **Chatterbox** *(Default Engine)* | **English (`en`)** *(US, UK, Canadian, Australian, Regional)* | **Zero-shot (10s audio)** | High-fidelity English voice cloning | MIT (Free / Commercial) |
| **Coqui XTTS-v2** *(Upgrade Option)* | **17 Languages**: Dutch (`nl`), Chinese (`zh`), English (`en`), Spanish (`es`), French (`fr`), German (`de`), Italian (`it`), Portuguese (`pt`), Polish (`pl`), Turkish (`tr`), Russian (`ru`), Arabic (`ar`), Czech (`cs`), Hungarian (`hu`), Korean (`ko`), Japanese (`ja`), Hindi (`hi`) | **Zero-shot (3–6s audio)** | Multilingual voice agents (e.g. Dutch + Chinese) | Open-Source (CPML / Hugging Face) |
| **CosyVoice 2/3** *(Upgrade Option)* | English, Chinese, Japanese, Korean, German, French, Spanish, Russian | **Zero-shot (3s audio)** | Ultra-low latency conversational AI (~150ms streaming) | Apache 2.0 (Open Source) |
| **Meta MMS** | **1,100+ Languages** worldwide | Fixed preset voices per language | Extreme dialect coverage (No custom voice cloning) | Apache 2.0 (Open Source) |

---

## 4. Hardware Requirements (CPU vs. GPU)

> [!IMPORTANT]  
> **Understanding Latency for Live Phone Calls:**  
> Vapi voice calls require audio generation within **1 to 2 seconds**. If synthesis takes longer than 5–10 seconds, Vapi terminates the call due to audio timeout.

| Hardware Tier | Real-Time Factor (RTF) | 1-Sentence Synthesis Time | Suitable For |
| :--- | :--- | :--- | :--- |
| **Local CPU (Intel i5/i7)** | ~8x – 12x | **25 – 45 seconds** | Local UI testing, voice verification, offline generation |
| **Standard VPS (e.g. Hostinger KVM 4, CPU-only)** | ~6x – 10x | **20 – 35 seconds** | Staging, API testing *(Too slow for live Vapi phone calls)* |
| **Dedicated GPU (NVIDIA T4 / RTX 3060 / A100)** | **<0.1x** | **0.2 – 0.4 seconds (Instant)** | **Production live conversational voice calls** |

*Cloud GPU instances (like RunPod, Vast.ai, or AWS G4dn) typically cost **$0.20 – $0.25/hour** (~$15–$25/month).*

---

## 5. Local Development Quickstart

### Prerequisites
- Python 3.11 installed (Python 3.12+ is not recommended due to PyTorch C-extensions)
- Git installed
- ~5 GB free disk space for neural model weights

### Setup Steps
```powershell
# 1. Clone or navigate to the repository
cd "C:\Users\noorr\OneDrive\Desktop\antogravity tests\Kokoro"

# 2. Configure Environment Secrets
Copy-Item .env.example .env

# Edit .env and set your secure secret token:
# VAPI_SECRET=vapi_sec_your_secret_here
```

### Running the Server
```powershell
# Windows PowerShell Start Script
.\start.ps1

# Or run directly with virtual environment:
.\venv311\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
When startup completes, you will see:
```text
INFO: Model loaded successfully. Native sample rate: 24000 Hz
INFO: Warmup complete in 22000 ms
INFO: Uvicorn running on http://127.0.0.1:8000
```

---

## 6. Testing the Engine

### Option A: Local Browser UI
Open your browser to:
👉 **[http://127.0.0.1:8000/preview/ui](http://127.0.0.1:8000/preview/ui)**

1. **Select Voice**: Pick between the standard default voice or any uploaded cloned voice.
2. **Synthesize**: Type any text and click `▶ Synthesize` to hear immediate audio playback.
3. **Upload New Voice**: Upload a 10–15s `.wav` or `.mp3` clip under *Voice Cloning* to add new custom voices on the fly.

### Option B: Automated Integration Test Client
Run the automated Vapi protocol test client to verify PCM 16-bit headers, byte stream format, and timing:
```powershell
.\venv311\Scripts\python.exe tests/test_client.py --text "Hello! Custom voice server is live." --rate 24000
```
This script saves the raw PCM output and wraps it in a playable WAV inside `tests/output/`.

---

## 7. Vapi Dashboard Configuration

Because Vapi runs in the cloud, expose your local machine via an HTTPS tunnel (e.g. `ngrok`):

```powershell
ngrok http 8000
```
Copy the generated forwarding URL (e.g., `https://xxxx-xx.ngrok-free.app`).

### Vapi Assistant Settings
In your **[Vapi Dashboard](https://dashboard.vapi.ai/)**:
1. Duplicate your assistant to create a safe test duplicate (e.g., *`Test Agent - Custom Voice`*).
2. Navigate to **Voice** settings.
3. Select **Provider**: `Custom Voice`.
4. Configure the parameters:
   - **Server URL**:
     ```text
     https://<your-subdomain>.ngrok-free.app/vapi/tts
     ```
   - **Voice ID**:
     Type the exact filename of your uploaded voice (e.g., `my_custom_voice.mp3`), or type `default` for the base voice.
     *(You can also use URL query parameters: `https://.../vapi/tts?voice=my_custom_voice.mp3`)*
   - **Custom Credentials / Secret Header**:
     - **Header Name**: `X-Vapi-Secret`
     - **Header Value**: `<VAPI_SECRET from your .env>`
   - **Sample Rate**: Select `24000` (or `16000`).
5. Click **Save** and start a test web/phone call.

---

## 8. Production Deployment Guide (Linux VPS / GPU Server)

To deploy on a remote Ubuntu/Debian server with a static public IP or domain:

### Step 1: System Packages
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip ffmpeg git nginx
```

### Step 2: Clone and Setup Virtual Environment
```bash
git clone <your-repo-url> /opt/vapi-tts
cd /opt/vapi-tts

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment
```bash
cp .env.example .env
nano .env
# Set VAPI_SECRET and ensure TTS_DEVICE=cuda if GPU is present
```

### Step 4: Systemd Service (24/7 Autostart)
Create a systemd unit file:
```bash
sudo nano /etc/systemd/system/vapi-tts.service
```
Paste the following:
```ini
[Unit]
Description=Vapi Custom Voice TTS Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/vapi-tts
ExecStart=/opt/vapi-tts/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
EnvironmentFile=/opt/vapi-tts/.env

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vapi-tts
sudo systemctl start vapi-tts
```

### Step 5: Nginx Reverse Proxy with SSL (HTTPS)
Configure Nginx to route traffic to port 8000 and enable Certbot SSL:
```nginx
server {
    server_name tts.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

---

## 9. API Reference & Webhook Contract

### `POST /vapi/tts`
The primary endpoint called by Vapi's webhook dispatcher.

- **Headers**:
  - `X-Vapi-Secret`: Required if authentication is enabled.
  - `Content-Type`: `application/json`
- **Query Parameters (Optional)**:
  - `?voice=<filename>`: Overrides voice selection directly via URL.
- **Request Body (Vapi Protocol)**:
  ```json
  {
    "message": {
      "type": "voice-request",
      "text": "Hello, thank you for calling. How can I help you today?",
      "sampleRate": 24000,
      "voiceId": "Recordonline-voice-recordercom1.mp3"
    }
  }
  ```
- **Response**:
  - **Status**: `200 OK`
  - **Content-Type**: `application/octet-stream`
  - **Body**: Raw 16-bit signed little-endian mono PCM audio.

---

### `GET /health`
Returns system status and model readiness.
```json
{
  "status": "ready",
  "model": "chatterbox-nano",
  "device": "cpu",
  "voice_profile": "client_default",
  "warmup_complete": true
}
```

---

### `GET /admin/voices`
Lists all available voices (default model voice + all uploaded audio files).

---

## 10. Troubleshooting & FAQ

#### Q: The call in Vapi connects, but then disconnects after a few seconds.
**Cause:** Your server is running on a CPU and taking ~30 seconds to generate the audio. Vapi drops calls if the voice is not delivered within 5–10 seconds.  
**Fix:** Deploy the server to an instance equipped with an NVIDIA GPU (RTX 3060, T4, A100). Response latency will drop to ~0.3 seconds.

#### Q: Vapi returns `503 Service Unavailable`.
**Cause:** The server is still warming up, or a second synthesis request arrived while the first was still computing on the CPU.  
**Fix:** Verify `/health` returns `{"status":"ready"}` before testing. On GPU, synthesis runs in 300ms, eliminating queue contention.

#### Q: How do I change the active voice without editing code?
**Fix:** Either select the voice in the **Select Voice** dropdown at `/preview/ui`, or specify the filename in Vapi's **Voice ID** field (e.g. `Recordonline-voice-recordercom1.mp3`).

---

## License & Credits
- **Server Framework:** MIT License
- **Chatterbox Model Weights:** Resemble AI (MIT License)
- **Audio Watermarking:** PerTh perceptual watermarking enabled by default per Resemble AI specifications.
