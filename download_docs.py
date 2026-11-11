"""Download 7 free research papers from arXiv into ./docs.  Run: python download_docs.py"""
import time
import urllib.request
from pathlib import Path

PAPERS = {
    "attention_is_all_you_need.pdf": "1706.03762",
    "bert.pdf": "1810.04805",
    "rag_retrieval_augmented_generation.pdf": "2005.11401",
    "gan_goodfellow.pdf": "1406.2661",
    "dcgan.pdf": "1511.06434",
    "resnet.pdf": "1512.03385",
    "word2vec.pdf": "1301.3781",
}

Path("docs").mkdir(exist_ok=True)
for name, arxiv_id in PAPERS.items():
    dest = Path("docs") / name
    if dest.exists():
        print(f"skip {name} (already there)")
        continue
    req = urllib.request.Request(
        f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "Mozilla/5.0"}
    )
    print(f"downloading {name} ...")
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    time.sleep(3)  # be polite to arXiv
print("Done. Now run: python ingest.py")
