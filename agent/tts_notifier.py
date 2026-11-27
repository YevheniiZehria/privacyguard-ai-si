"""Local, offline, CPU-only text-to-speech notifications.

Order of preference: pyttsx3 (Windows SAPI5 / espeak) -> piper CLI -> console print.
No audio ever leaves the machine.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile


def _speak_pyttsx3(text: str) -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True
    except Exception:
        return False


def _speak_piper(text: str) -> bool:
    exe = os.getenv("PIPER_EXE") or shutil.which("piper")
    voice = os.getenv("PIPER_VOICE")
    if not exe or not voice:
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "tts.wav")
            subprocess.run([exe, "--model", voice, "--output_file", wav], input=text.encode("utf-8"),
                           check=True, capture_output=True, timeout=60)
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(wav, winsound.SND_FILENAME)
            else:
                player = shutil.which("aplay") or shutil.which("afplay")
                if player:
                    subprocess.run([player, wav], check=False, capture_output=True)
        return True
    except Exception:
        return False


def build_message(stats: dict) -> str:
    docs = stats.get("documents_processed", 0)
    boxes = stats.get("boxes_redacted", 0)
    passed = stats.get("verified_pass", 0)
    failed = stats.get("verified_fail", 0)
    clusters = stats.get("clusters", 0)
    linked = stats.get("linked_documents", 0)
    status = stats.get("status", "completed")
    parts = [f"Task {status}.", f"{docs} document{'s' if docs != 1 else ''} processed,",
             f"{boxes} PII region{'s' if boxes != 1 else ''} redacted."]
    if passed or failed:
        parts.append(f"Verification: {passed} passed, {failed} failed.")
    if clusters:
        parts.append(f"{clusters} document link group{'s' if clusters != 1 else ''} found, covering {linked} documents.")
    else:
        parts.append("No document links found.")
    return " ".join(parts)


def announce(stats: dict, enabled: bool = True) -> str:
    message = build_message(stats)
    print(f"\n[TTS] {message}")
    if enabled and os.getenv("TTS_ENABLED", "true").lower() not in ("0", "false", "no"):
        if not _speak_pyttsx3(message):
            _speak_piper(message)
    return message


if __name__ == "__main__":
    announce({"documents_processed": 3, "boxes_redacted": 17, "verified_pass": 3, "clusters": 1,
              "linked_documents": 3})
