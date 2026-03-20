#!/usr/bin/env python3
"""
Fast Whisper Dictation — GPU-accelerated push-to-talk for Linux/X11.

Hold the trigger key to record, release to transcribe and paste.
Model stays loaded in VRAM for instant responses.
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
import warnings

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000

FORMAT_PROMPT = """Clean up dictated text. Keep the speaker's exact words. Only fix punctuation, capitalization, and obvious self-corrections. Do not rephrase or reword anything. When the speaker dictates structure like "new paragraph", "bullet point", "dash", "number one", or "next item", produce the corresponding formatting. Also detect implicit structure: ordinals like "firstly/second/third" become numbered lists, and enumerated items after introductory phrases become bulleted lists.

When the speaker restarts a sentence — saying nearly the same thing again with slightly different words — keep only the final version. Look for back-to-back phrases that share the same opening words or structure, where the second one is clearly a second attempt. Only do this when the overlap is obvious; if the repetition looks intentional (e.g. for emphasis or listing), keep both.

---

Input: come over at three I mean four oclock
Output: Come over at 4 o'clock.
---

Input: I need to buy eggs milk no wait not milk butter and bread
Output: I need to buy eggs, butter, and bread.
---

Input: add the user to the admin group sorry I mean the editors group
Output: Add the user to the editors group.
---

Input: we should commit and push well not push the changes we have now
Output: We should commit the changes we have now.
---

Input: lets add logging and metrics well maybe not metrics for now to the service
Output: Let's add logging to the service.
---

Input: I need to update and deploy well not deploy yet the new config
Output: I need to update the new config.
---

Input: we should refactor and rewrite well maybe not rewrite just clean up the module
Output: We should refactor and just clean up the module.
---

Input: I'll just read some stuff back to them I'll just read some stuff back to you then shall I
Output: I'll just read some stuff back to you then, shall I?
---

Input: we should probably set up we should set up a staging environment first
Output: We should set up a staging environment first.
---

Input: the thing is the thing is that nobody actually uses this feature
Output: The thing is that nobody actually uses this feature.
---

Input: we need ten no twenty servers for this
Output: We need 10... no, 20 servers for this.

---

Input: what should we choose JSON or YAML
Output: What should we choose, JSON or YAML?
---

Input: send it to john at example dot com
Output: Send it to john@example.com.
---

Input: the meeting is on tuesday actually wait its wednesday at three pm
Output: The meeting is on Tuesday... actually wait, it's Wednesday at 3 PM.
---

Input: so basically I think we should probably just go with the simpler approach
Output: So basically I think we should probably just go with the simpler approach.
---

Input: first we need to check the logs new paragraph then once we have the error we can start debugging new paragraph finally we should add a test so this doesnt happen again
Output: First we need to check the logs.

Then once we have the error we can start debugging.

Finally we should add a test so this doesn't happen again.
---

Input: things we need to do dash update the database dash fix the login bug dash deploy to staging
Output: Things we need to do:
- Update the database
- Fix the login bug
- Deploy to staging
---

Input: the steps are number one clone the repo number two install dependencies number three run the tests
Output: The steps are:
1. Clone the repo
2. Install dependencies
3. Run the tests
---

Input: we need to implement the following things firstly update the database second fix the login and third deploy to staging
Output: We need to implement the following things:
1. Update the database
2. Fix the login
3. Deploy to staging
---

Input: we need to get the following items from the shop sausages milk bread cheese ice
Output: We need to get the following items from the shop:
- Sausages
- Milk
- Bread
- Cheese
- Ice
---

Input: {text}
Output:"""


class Dictation:
    def __init__(self, model_size, language, device, compute_type, format_model=None):
        print(f"Loading whisper model '{model_size}' on {device} ({compute_type})...")
        t0 = time.time()
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type,
            cpu_threads=4,
        )
        print(f"Whisper model loaded in {time.time() - t0:.1f}s")

        self.llm = None
        if format_model:
            from llama_cpp import Llama
            print(f"Loading format model '{os.path.basename(format_model)}'...")
            t0 = time.time()
            self.llm = Llama(
                model_path=format_model, n_gpu_layers=-1,
                n_ctx=4096, verbose=False,
            )
            print(f"Format model loaded in {time.time() - t0:.1f}s")

        self.language = language
        self.recording = False
        self.audio_chunks = []
        self.stream = None
        self._lock = threading.Lock()

    def start_recording(self):
        with self._lock:
            if self.recording:
                return
            self.recording = True
            self.audio_chunks = []
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
        # Subtle notification
        subprocess.Popen(
            ["notify-send", "-t", "800", "-u", "low", "Dictation", "Recording..."],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("\033[91m● REC\033[0m", end=" ", flush=True)

    def _audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_chunks.append(indata.copy())

    def stop_and_transcribe(self):
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            chunks = self.audio_chunks
            self.audio_chunks = []

        if not chunks:
            print("(empty)")
            return

        audio = np.concatenate(chunks, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE

        if duration < 0.3:
            print("(too short)")
            return

        t0 = time.time()
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.time() - t0

        if text:
            if self.llm:
                t1 = time.time()
                text = self._format(text)
                fmt_elapsed = time.time() - t1
                print(f"\033[92m✓\033[0m [{elapsed:.2f}s whisper + {fmt_elapsed:.2f}s fmt / {duration:.1f}s audio] {text}")
            else:
                print(f"\033[92m✓\033[0m [{elapsed:.2f}s / {duration:.1f}s audio] {text}")
            self._paste(text)
        else:
            print("(no speech detected)")

    def _format(self, text):
        prompt = FORMAT_PROMPT.format(text=text)
        try:
            out = self.llm(prompt, max_tokens=max(len(text) * 2, 200),
                           temperature=0, stop=["---", "Input:"])
            result = out["choices"][0]["text"]
            # Strip thinking tags from models that emit them (e.g. Qwen3.5)
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            return result if result else text
        except Exception as e:
            print(f"(format error: {e})")
            return text

    def _paste(self, text):
        # Copy to clipboard
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
        )
        proc.communicate(text.encode("utf-8"))

        # Small delay to ensure clipboard is set
        time.sleep(0.02)

        # Ctrl+Shift+V works universally:
        # - Terminals (Ghostty, etc.): clipboard paste
        # - Browsers (Chrome, Firefox): paste as plain text
        # - Web-based terminals in browsers: clipboard paste
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def parse_key(key_name):
    """Map a user-friendly key name to pynput key attributes."""
    key_map = {
        "alt_r": keyboard.Key.alt_r,
        "super_r": keyboard.Key.cmd_r,
        "super_l": keyboard.Key.cmd_l,
        "super": keyboard.Key.cmd,
        "caps_lock": keyboard.Key.caps_lock,
        "scroll_lock": keyboard.Key.scroll_lock,
        "pause": keyboard.Key.pause,
        "insert": keyboard.Key.insert,
        "f1": keyboard.Key.f1, "f2": keyboard.Key.f2,
        "f3": keyboard.Key.f3, "f4": keyboard.Key.f4,
        "f5": keyboard.Key.f5, "f6": keyboard.Key.f6,
        "f7": keyboard.Key.f7, "f8": keyboard.Key.f8,
        "f9": keyboard.Key.f9, "f10": keyboard.Key.f10,
        "f11": keyboard.Key.f11, "f12": keyboard.Key.f12,
    }
    k = key_name.lower().replace("-", "_").replace(" ", "_")
    if k in key_map:
        return key_map[k]
    # Try as a character key
    if len(k) == 1:
        return keyboard.KeyCode.from_char(k)
    print(f"Unknown key '{key_name}'. Available: {', '.join(key_map.keys())}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Fast Whisper Dictation — push-to-talk for Linux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s                          # defaults: large-v3-turbo, English, right Super key
  %(prog)s --model large-v3         # max accuracy
  %(prog)s --model small.en         # lighter, faster, English-only
  %(prog)s --key caps_lock          # use Caps Lock as trigger
  %(prog)s --key f8                 # use F8 as trigger
  %(prog)s --language auto          # auto-detect language
""",
    )
    parser.add_argument(
        "--model", default="large-v3-turbo",
        help="Whisper model (default: large-v3-turbo). Options: tiny, base, small, "
             "medium, large-v3, large-v3-turbo. Add .en suffix for English-only variants.",
    )
    parser.add_argument("--language", default="en", help="Language code or 'auto' (default: en)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--compute-type", default="float16",
        help="Compute type (default: float16). Options: float16, int8_float16, int8, float32",
    )
    parser.add_argument(
        "--key", default="super_r",
        help="Push-to-talk key (default: super_r). Examples: super_r, caps_lock, f8, pause",
    )
    parser.add_argument(
        "--no-format", action="store_true",
        help="Disable LLM formatting pass (raw whisper output)",
    )
    parser.add_argument(
        "--format-model", default=None,
        help="Path to GGUF model for formatting. Auto-detected if not set.",
    )
    args = parser.parse_args()

    lang = None if args.language == "auto" else args.language
    trigger = parse_key(args.key)

    # Auto-detect format model (prefer newer/smaller models first)
    fmt_model = None
    if not args.no_format:
        if args.format_model:
            fmt_model = args.format_model
        else:
            model_dir = os.path.expanduser("~/.local/share/whisper-dictation/models")
            for name in [
                "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                "Qwen3.5-4B-Q4_K_M.gguf",
                "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            ]:
                path = os.path.join(model_dir, name)
                if os.path.exists(path):
                    fmt_model = path
                    break

    dictation = Dictation(args.model, lang, args.device, args.compute_type, fmt_model)

    print(f"\nReady! Hold [{args.key}] to record, release to transcribe.")
    print("Press Ctrl+C to quit.\n")

    def on_press(key):
        if key == trigger:
            dictation.start_recording()

    def on_release(key):
        if key == trigger:
            threading.Thread(
                target=dictation.stop_and_transcribe, daemon=True
            ).start()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\nBye!")


if __name__ == "__main__":
    main()
