"""Run the bot on eval_questions.json and write eval_results.csv.

Run:  python evaluate.py

Step 1 (automatic): checks whether the answer contains the expected keywords.
Step 2 (manual, important): open eval_results.csv, read each answer, and fill the
'manual_correct' column with 1 or 0. Report THAT number on your resume, since
keyword matching can miss correct paraphrases or accept wrong answers.
"""
import csv
import json
import time

from rag import answer, get_llm, load_store

QUESTIONS_FILE = "eval_questions.json"
OUTPUT_FILE = "eval_results.csv"
DELAY_SECONDS = 4  # keeps you under free-tier rate limits


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    store, llm = load_store(), get_llm()
    rows, auto_correct = [], 0

    for i, q in enumerate(questions, start=1):
        text, docs = answer(q["question"], store, llm)
        keywords = [k.lower() for k in q["expected_keywords"]]
        hit = all(k in text.lower() for k in keywords)
        auto_correct += hit
        sources = "; ".join(
            f"{d.metadata.get('source_name')} p.{d.metadata.get('page_label')}"
            for d in docs
        )
        rows.append(
            {
                "id": i,
                "question": q["question"],
                "expected_keywords": ", ".join(q["expected_keywords"]),
                "answer": text,
                "sources": sources,
                "auto_correct": int(hit),
                "manual_correct": "",
            }
        )
        print(f"[{i}/{len(questions)}] auto={'OK ' if hit else 'MISS'} | {q['question']}")
        time.sleep(DELAY_SECONDS)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAutomatic keyword score: {auto_correct}/{len(questions)}")
    print(f"Now open {OUTPUT_FILE} and fill 'manual_correct' (1/0) for the real score.")


if __name__ == "__main__":
    main()
