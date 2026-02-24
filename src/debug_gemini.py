import os
import google.generativeai as genai
from dotenv import load_dotenv
import traceback

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

print(f"Checking API key: {api_key[:5]}...{api_key[-5:]}")

try:
    print("\nListing all available models:")
    models = genai.list_models()
    count = 0
    for m in models:
        count += 1
        methods = ", ".join(m.supported_generation_methods)
        print(f"- {m.name} (Methods: {methods})")
    
    if count == 0:
        print("❓ No models found.")
except Exception as e:
    print("❌ Error listing models:")
    print(str(e))
    traceback.print_exc()
