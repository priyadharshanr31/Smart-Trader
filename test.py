import os, requests
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("GEMINI_API_KEY")
resp = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": key})
print(resp.status_code, resp.json())
