# 🎙️ Standalone Real-Time Voice Transcription & Speaker Biometrics

A self-contained, high-performance real-time speech transcription and local biometric speaker recognition system.

---

## ✨ Features

- **Multi-STT Provider Router:**
  - 🥇 **Sarvam Saarika v2.5** (Indian English, Hindi, Tamil & Indic SOTA)
  - 🥈 **OpenAI gpt-live-transcribe / Whisper** (Cloud transcription)
  - 🥉 **Faster-Whisper** (Local CPU transcription fallback)
- **Local ECAPA-TDNN Biometrics:**
  - 192-dimensional speaker voiceprint embeddings on CPU.
  - **Hybrid Dual-Scoring Matcher:** Instant vector matching against Master Centroid + individual enrolled voice samples.
  - **Temporal Continuity Smoothing:** Eliminates single-chunk dropouts on short/soft speech turns.
- **ChatGPT-Style Voice Chat Interface:**
  - In-place real-time streaming dialogue cards.
  - Interactive voice sample enrollment modal.
  - Model selector dropdown.

---

## 🚀 Quickstart

### 1. Local Python Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment keys
cp .env.example .env
# Add your SARVAM_API_KEY / OPENAI_API_KEY in .env

# 4. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Docker Setup

```bash
docker compose up --build
```

---

## 🌐 Endpoints

* 💬 **Voice Chat UI:** [http://localhost:8000/voice-chat](http://localhost:8000/voice-chat)
* 📖 **API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
* 💓 **Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)
