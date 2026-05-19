import json
from collections import Counter

scopes = Counter()
with open("rag_project/data/raw/master_lich_tuan.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        scope = rec.get("metadata", {}).get("scope", "(none)")
        scopes[scope] += 1

total = sum(scopes.values())
print(f"Total records: {total}")
for scope, count in scopes.most_common():
    pct = count / total * 100
    print(f"  [{scope}]: {count} ({pct:.1f}%)")
