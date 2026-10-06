"""One-time migration: fix Q-dial cost_overhead values produced by the old
(Q*gamma)^(2k) formula. Corrected model: retained overhead ~= max(1, Q * gamma^(2k)),
where gamma^(2k) is the curve's own Q=1.0 cost (formula-independent at Q=1).
Idempotent: skips curves already migrated (marker key).  Run: python scripts/migrate_qdial_cost.py
"""
import json

for path in ["results/h1_results.json", "results/h2_results.json"]:
    d = json.load(open(path))
    n = 0
    for row in d:
        curve = row.get("Qdial")
        if not curve or row.get("_qdial_cost_migrated"):
            continue
        cost_q1 = next(p["cost_overhead"] for p in curve if p["Q"] >= 1.0)
        for p in curve:
            p["cost_overhead"] = max(1.0, p["Q"] * cost_q1)
        row["_qdial_cost_migrated"] = True
        n += 1
    json.dump(d, open(path, "w"), indent=2)
    print(f"{path}: migrated {n} Qdial curves")
