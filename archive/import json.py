import json
from pathlib import Path
from collections import Counter

p = Path(r"D:\Annual report LLM project\LLM project_20251207\outputs\extractions_2024.jsonl")

cnt = Counter()
bad = []
total = 0
with p.open("r", encoding="utf-8") as f:
    for line in f:
        total += 1
        obj = json.loads(line)
        bank = obj.get("_meta", {}).get("bank", "UNKNOWN")
        results = obj.get("results", None)
        if isinstance(results, list):
            cnt[len(results)] += 1
        else:
            cnt["NOT_LIST"] += 1
            bad.append(bank)

print("records:", total)
print("results length distribution:", cnt)
if bad:
    print("banks with non-list results:", bad[:10], "...")
