# VoiceLab — Privacy

## Scope

This document describes how VoiceLab handles audio data, text input, and voice cloning samples. VoiceLab is a self-hosted tool; the operator (the person running the server) controls all data flows.

---

## Audio Data Handling

### Synthesized audio

Synthesized audio is never stored persistently by VoiceLab itself.

**Gradio UI path:**
- WAV output (LocalTts): decoded to a numpy array in memory; no file is written
- MP3 output (ElevenLabsTts): written to a unique `NamedTemporaryFile` in `/tmp` (e.g. `/tmp/voicelab_out_<uuid>.mp3`) for Gradio to serve; the file is not deleted by VoiceLab after serving

**REST API path (`POST /api/v1/synthesize`):**
- Audio bytes are streamed directly to the HTTP response as `audio/mpeg`
- No file is written to disk; nothing is retained after the response completes

**REST API path (`POST /api/v1/compare`):**
- Both audio results are base64-encoded and returned in a JSON response body
- No files are written; nothing is retained after the response completes

**Temp file lifecycle (`/tmp`):**
- Files in `/tmp` are not cleaned up by VoiceLab
- In Docker, `/tmp` is an in-container ephemeral filesystem; it is erased when the container stops
- On a persistent host, use OS-level temp file cleanup (e.g. `systemd-tmpfiles`, `tmpwatch`, or `tmpfs` mount)

### ElevenLabs cloud synthesis

When `ELEVENLABS_API_KEY` is set:
- The text to synthesize and the selected voice_id are sent to the ElevenLabs API over HTTPS
- ElevenLabs processes this data under their own terms of service and privacy policy: https://elevenlabs.io/privacy
- VoiceLab does not cache or log the content of synthesis requests

---

## Biometric Data Notice

**Voice is biometric data.** A voice sample uniquely identifies a person and is classified as sensitive personal data under GDPR, CCPA, and similar frameworks.

### Voice cloning

The Voice Cloning tab and `POST /api/v1/clone` endpoint:
1. Accept an uploaded audio sample (WAV / MP3 / OGG / M4A / FLAC)
2. Write the sample to a temp file in `/tmp` for validation
3. Send the sample to the ElevenLabs API to create a cloned voice model

**When cloning is invoked:**
- The audio sample (biometric data) leaves VoiceLab and is transmitted to ElevenLabs
- ElevenLabs stores the voice model on their infrastructure
- VoiceLab retains the temp file in `/tmp` only for the duration of the request; it is not stored elsewhere

**Data subject rights:** If you need to delete a cloned voice, do so via the ElevenLabs dashboard or API. VoiceLab does not provide a deletion mechanism.

### Consent requirement

The `/api/v1/clone` endpoint requires `consent=true` in every request. This is a technical gate — it does not replace the operator's obligation to obtain lawful consent from the voice owner. Operators must:

1. Obtain explicit, informed consent from the person whose voice is being cloned
2. Inform them that their biometric voice data will be sent to ElevenLabs and retained by ElevenLabs
3. Comply with applicable biometric privacy laws (BIPA in Illinois, GDPR Article 9 in the EU, etc.)

---

## Keyless Mode

When `ELEVENLABS_API_KEY` is not set:
- No data leaves the process
- Synthesis uses the local pyttsx3 system TTS engine
- Voice Cloning is unavailable
- All audio is generated and served locally

Keyless mode is the default and the privacy-preserving option for development and offline use.

---

## Text Input

Text submitted for synthesis is:
- In keyless mode: processed entirely within the local process; not logged
- In cloud mode: sent to ElevenLabs over HTTPS as part of the synthesis request

VoiceLab logs synthesis calls at INFO level (outcome and latency) but does not log the text content.

---

## API Keys

`ELEVENLABS_API_KEY` is:
- Read from environment variables or `.env` file
- Filtered from all log output by `SecretRedactionFilter`
- Never returned in API responses

See [SECURITY.md](SECURITY.md) for full details.

---

## External Links

- ElevenLabs Privacy Policy: https://elevenlabs.io/privacy
- ElevenLabs Terms of Service: https://elevenlabs.io/terms
