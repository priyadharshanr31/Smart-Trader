# test_gemini_qa.py
"""
Simple script to test your GEMINI_API_KEY by asking a question.

- Loads GEMINI_API_KEY from .env (or environment).
- Sends your question to Gemini.
- Prints the raw answer.

Usage:
  python test_gemini_qa.py
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env (if present)
load_dotenv()

# Optional: put a key here for one-off testing (otherwise it uses .env)
EXPLICIT_KEY = None  # e.g. "AIzaSy..." – leave as None to use GEMINI_API_KEY from .env


def get_api_key() -> str:
    key = (EXPLICIT_KEY or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in your .env or EXPLICIT_KEY."
        )
    return key


def main():
    # 1) Get key and configure client
    key = get_api_key()
    print("Using GEMINI_API_KEY that starts with:", key[:10], "…")
    genai.configure(api_key=key)

    # 2) Ask user for a question
    question = input("\nEnter your question for Gemini:\n> ").strip()
    if not question:
        print("No question entered, exiting.")
        return

    # 3) Call Gemini
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(question)

        # Prefer resp.text, fall back to first candidate text if needed
        text = getattr(response, "text", None)
        if not text and getattr(response, "candidates", None):
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception:
                text = None

        if not text:
            print("\n❌ Got an empty response from Gemini.")
        else:
            print("\n✅ Gemini response:\n")
            print(text)
    except Exception as e:
        print("\n❌ Gemini call FAILED:")
        print(type(e).__name__, ":", e)


if __name__ == "__main__":
    main()
