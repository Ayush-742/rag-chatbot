import os, time, sys
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={"timeout": 60_000},  # milliseconds: give up after 60 seconds
)

models = sys.argv[1:]
for name in models:
    t = time.time()
    try:
        r = client.models.generate_content(model=name, contents="Reply with the single word OK.")
        print(f"{name}: {time.time() - t:.1f}s -> {r.text.strip()!r}")
    except Exception as e:
        print(f"{name}: {time.time() - t:.1f}s -> ERROR {str(e)[:150]}")