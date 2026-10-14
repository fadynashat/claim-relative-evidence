#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible controlled cases A-I for the EWDA/P1 evidence model.

Implements the four models in the manuscript appendix and checks numerical
references, boundaries, malformed inputs, classifications, deletion bounds,
and removal sensitivity across all cases.
Run: python3 controlled_cases_data_hardened.py
"""
from __future__ import annotations
import math
from typing import Any, Optional

K = 3
CRITICAL_DIMENSIONS = (0, 1)  # zero-based: q_1 and q_2
TAU_C, TAU_W = 0.30, 0.40
THETA_MINUS, THETA_PLUS = -0.20, 0.20
MODELS = ("M0", "M1", "M1'", "M2")
STATUSES = {"ACCEPTED", "REJECTED", "PENDING"}
Aggregate = Optional[float]


# ============================================================
# 1. Core validation
# ============================================================

def valid_number(x: Any, lo: float, hi: float) -> bool:
    return (not isinstance(x, bool) and isinstance(x, (int, float))
            and math.isfinite(x) and lo <= x <= hi)


def valid_q(q: Any) -> bool:
    """Accept both tuple and list for Q, matching the mathematical domain."""
    return (isinstance(q, (tuple, list)) and len(q) == K
            and all(valid_number(x, 0.0, 1.0) for x in q))


# ============================================================
# 2. Model functions
# ============================================================

def raw_weight(q: tuple[float, ...]) -> float:
    """Geometric mean; validates its own domain rather than trusting callers."""
    if not valid_q(q):
        raise ValueError(f"Q must be a tuple of {K} finite values in [0,1]")
    return math.prod(q) ** (1.0 / K)


def numerical_eligibility(q: tuple[float, ...]) -> int:
    if not valid_q(q):
        raise ValueError(f"Q must be a tuple of {K} finite values in [0,1]")
    if any(q[i] < TAU_C for i in CRITICAL_DIMENSIONS):
        return 0
    if max(q) < TAU_W:
        return 0
    return int(raw_weight(q) > 0.0)


def validate_inputs(items: Any) -> bool:
    if not isinstance(items, list):
        return False
    required = {"Q", "s", "status"}
    for item in items:
        if not isinstance(item, dict) or not required.issubset(item):
            return False
        if not valid_q(item["Q"]):
            return False
        if not valid_number(item["s"], -1.0, 1.0):
            return False
        if not isinstance(item["status"], str) or item["status"] not in STATUSES:
            return False
    return True


def contribution(item: dict, model: str) -> Optional[tuple[float, float]]:
    q, s, status = item["Q"], item["s"], item["status"]
    if model == "M0":
        return 1.0, float(s)
    if model == "M1":
        return raw_weight(q), float(s)
    if model == "M1'":
        return (raw_weight(q), float(s)) if status == "ACCEPTED" else None
    if model == "M2":
        return (raw_weight(q), float(s)) if status == "ACCEPTED" and numerical_eligibility(q) else None
    raise ValueError(f"Unknown model: {model}")


def aggregate(items: list[dict], model: str) -> Aggregate:
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if not validate_inputs(items):
        raise ValueError("Invalid evidence inputs")
    numerator = denominator = 0.0
    for item in items:
        part = contribution(item, model)
        if part is None:
            continue
        w, s = part
        numerator += w * s
        denominator += w
    return None if denominator == 0.0 else numerator / denominator


def classify(value: Aggregate) -> str:
    if value is None:
        return "Abstain"
    if value >= THETA_PLUS:
        return "Supported"
    if value <= THETA_MINUS:
        return "Contradicted"
    return "Undetermined"


def fmt(value: Aggregate, places: int = 4) -> str:
    return "⊥" if value is None else f"{value:.{places}f}"


# ============================================================
# 3. Test data
# ============================================================

CASES = {
 "A": [{"Q":(0.9,0.9,0.8),"s":0.8,"status":"ACCEPTED"},{"Q":(0.8,0.8,0.9),"s":0.6,"status":"ACCEPTED"}],
 "B": [{"Q":(0.9,0.9,0.8),"s":0.8,"status":"ACCEPTED"},{"Q":(0.2,0.9,0.8),"s":-0.9,"status":"ACCEPTED"}],
 "C": [{"Q":(0.9,0.9,0.8),"s":0.8,"status":"ACCEPTED"},{"Q":(0.9,0.25,0.9),"s":0.9,"status":"ACCEPTED"}],
 "D": [{"Q":(0.9,0.9,0.8),"s":0.8,"status":"ACCEPTED"},{"Q":(0.8,0.8,0.9),"s":-0.8,"status":"ACCEPTED"}],
 "E": [{"Q":(0.2,0.9,0.8),"s":0.8,"status":"ACCEPTED"},{"Q":(0.1,0.1,0.1),"s":-0.9,"status":"ACCEPTED"}],
 "F": [{"Q":(0.31,0.9,0.8),"s":0.8,"status":"ACCEPTED"},{"Q":(0.29,0.9,0.8),"s":0.7,"status":"ACCEPTED"}],
 "G": [{"Q":(0.9,0.9,0.8),"s":0.8,"status":"ACCEPTED"},{"Q":(0.8,0.8,0.9),"s":0.6,"status":"ACCEPTED"}],
 "H": [{"Q":(0.9,0.9,0.8),"s":0.8,"status":"ACCEPTED"},{"Q":(0.9,0.9,0.3),"s":0.7,"status":"ACCEPTED"}],
 "I": [{"Q":(0.9,0.9,0.8),"s":0.8,"status":"ACCEPTED"},{"Q":(0.8,0.8,0.9),"s":0.6,"status":"REJECTED"}],
}

EXPECTED = {
 "A":{"M0":.7000,"M1":.7020,"M1'":.7020,"M2":.7020},
 "B":{"M0":-.0500,"M1":.1587,"M1'":.1587,"M2":.8000},
 "C":{"M0":.8500,"M1":.8404,"M1'":.8404,"M2":.8000},
 "D":{"M0":.0000,"M1":.0157,"M1'":.0157,"M2":.0157},
 "E":{"M0":-.0500,"M1":.5276,"M1'":.5276,"M2":None},
 "F":{"M0":.7500,"M1":.7506,"M1'":.7506,"M2":.8000},
 "G":{"M0":.7000,"M1":.7020,"M1'":.7020,"M2":.7020},
 "H":{"M0":.7500,"M1":.7581,"M1'":.7581,"M2":.7581},
 "I":{"M0":.7000,"M1":.7020,"M1'":.8000,"M2":.8000},
}

# High-precision reference values for raw weights
RAW_WEIGHT_REFERENCES = [
    ((0.9, 0.9, 0.8), 0.8653497421844450),
    ((0.8, 0.8, 0.9), 0.8320335292207616),
    ((0.2, 0.9, 0.8), 0.5241482788417793),
    ((0.9, 0.25, 0.9), 0.5872301461753295),
    ((0.1, 0.1, 0.1), 0.1000000000000000),
    ((0.31, 0.9, 0.8), 0.6065939345778796),
    ((0.29, 0.9, 0.8), 0.5932578562689901),
    ((0.9, 0.9, 0.3), 0.6240251469155712),
]


# ============================================================
# 4. Removal sensitivity
# ============================================================

def removal_sensitivity(items: list[dict], index: int) -> dict[str, dict[str, Aggregate]]:
    if not validate_inputs(items):
        raise ValueError("Invalid evidence inputs")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if not 0 <= index < len(items):
        raise IndexError("index out of range")
    reduced = items[:index] + items[index+1:]
    result = {}
    for model in MODELS:
        before, after = aggregate(items, model), aggregate(reduced, model)
        parts = [contribution(item, model) for item in items]
        total = sum(p[0] for p in parts if p is not None)
        removed = parts[index]
        if before is None or after is None:
            delta = bound = None
        else:
            delta = abs(before-after)
            bound = 0.0 if removed is None or total == 0 else 2.0*removed[0]/total
        result[model] = {"before":before,"after":after,"delta":delta,"bound":bound}
    return result


def close(actual: Aggregate, expected: Aggregate, tol: float=5e-4) -> bool:
    if expected is None:
        return actual is None
    return actual is not None and abs(actual-expected) < tol


# ============================================================
# 5. All-cases removal sensitivity check
# ============================================================

def check_removal_all_cases() -> tuple[bool, str]:
    """Test removal bound across all cases and all removable items."""
    failures = []
    total_tests = 0
    for case_name, items in CASES.items():
        if len(items) < 2:
            continue
        for model in MODELS:
            s_full = aggregate(items, model)
            if s_full is None:
                continue
            for i in range(len(items)):
                reduced = items[:i] + items[i+1:]
                s_reduced = aggregate(reduced, model)
                if s_reduced is None:
                    continue
                total_tests += 1
                parts = [contribution(item, model) for item in items]
                total_w = sum(p[0] for p in parts if p is not None)
                removed = parts[i]
                if removed is None or total_w == 0:
                    continue
                bound = 2.0 * removed[0] / total_w
                delta = abs(s_full - s_reduced)
                if delta > bound + 1e-12:
                    failures.append(
                        f"{case_name}|{model}|remove e{i+1}: "
                        f"|Δ|={delta:.6f} > bound={bound:.6f}"
                    )
    if failures:
        return False, f"FAIL ({len(failures)} of {total_tests}): " + "; ".join(failures[:3])
    return True, f"PASS (all {total_tests} removal tests)"


# ============================================================
# 6. High-precision raw weight check
# ============================================================

def check_raw_weights_exact() -> tuple[bool, str]:
    """Verify raw weights at full double precision (tolerance 1e-12)."""
    failures = []
    for q, expected in RAW_WEIGHT_REFERENCES:
        got = raw_weight(q)
        if abs(got - expected) >= 1e-12:
            failures.append(f"{q}: got {got:.15f}, expected {expected:.15f}")
    if failures:
        return False, "FAIL: " + "; ".join(failures)
    return True, f"PASS ({len(RAW_WEIGHT_REFERENCES)} exact values)"


# ============================================================
# 7. List-acceptance check
# ============================================================

def check_list_acceptance() -> tuple[bool, str]:
    """Verify that Q accepts both tuple and list."""
    list_case = [{"Q":[0.9,0.9,0.8],"s":0.8,"status":"ACCEPTED"},
                 {"Q":[0.8,0.8,0.9],"s":0.6,"status":"ACCEPTED"}]
    if not validate_inputs(list_case):
        return False, "FAIL: list-Q rejected"
    try:
        agg = aggregate(list_case, "M0")
        if agg is None or abs(agg - 0.7) > 5e-4:
            return False, f"FAIL: list-Q aggregate wrong: {agg}"
    except Exception as e:
        return False, f"FAIL: list-Q raised {e}"
    return True, "PASS (list Q accepted)"


# ============================================================
# 8. Main check runner
# ============================================================

def run_checks() -> bool:
    checks: list[tuple[str, bool, str]] = []

    # 1. Valid / malformed inputs
    valid_ok = all(validate_inputs(v) for v in CASES.values())
    malformed = [
        None, {}, [None], [{}],
        [{"Q":(1,2),"s":0,"status":"ACCEPTED"}],
        [{"Q":(1,1,True),"s":0,"status":"ACCEPTED"}],
        [{"Q":(1,1,float('nan')),"s":0,"status":"ACCEPTED"}],
        [{"Q":(1,1,float('inf')),"s":0,"status":"ACCEPTED"}],
        [{"Q":(1,1,1),"s":True,"status":"ACCEPTED"}],
        [{"Q":(1,1,1),"s":0,"status":None}],
        [{"Q":(1,1,1),"status":"ACCEPTED"}],
        [{"Q":(1,1,1),"s":0,"status":"UNKNOWN"}],
        [{"Q":(1,1,-0.1),"s":0,"status":"ACCEPTED"}],
        [{"Q":(1,1,1.5),"s":0,"status":"ACCEPTED"}],
        [{"Q":(1,1,1),"s":1.5,"status":"ACCEPTED"}],
        [{"Q":(1,1,1),"s":-1.5,"status":"ACCEPTED"}],
        [{"Q":(1,1,float('-inf')),"s":0,"status":"ACCEPTED"}],
        [{"Q":(1,1,1),"s":float('nan'),"status":"ACCEPTED"}],
        [{"Q":(1,1,1),"s":float('inf'),"status":"ACCEPTED"}],
    ]
    malformed_ok = all(not validate_inputs(x) for x in malformed)
    checks.append(("valid cases and malformed input", valid_ok and malformed_ok,
                   f"{'PASS' if valid_ok and malformed_ok else 'FAIL'}"))

    # 2. Raw weights
    weight_ok = all(abs(raw_weight(q)-e) < 1e-9 for q, e in RAW_WEIGHT_REFERENCES)
    for q in [(), (.2, .3), (.2, .3, 1.2), (.2, .3, float('nan'))]:
        try:
            raw_weight(q)
            weight_ok = False
        except ValueError:
            pass
    checks.append(("raw weights and domain guards", weight_ok,
                   "PASS" if weight_ok else "FAIL"))

    # 3. Eligibility
    elig = [((.9,.9,.8),1),((.2,.9,.8),0),((.9,.25,.9),0),((.1,.1,.1),0),
            ((.31,.9,.8),1),((.29,.9,.8),0),((.9,.9,.3),1),((0,.9,.9),0),
            ((.30,.9,.8),1),((.9,.30,.8),1),((.9,.9,.40),1),
            ((.35,.35,.35),0),((.40,.30,.30),1)]
    elig_ok = all(numerical_eligibility(q) == e for q, e in elig)
    checks.append(("eligibility including boundaries", elig_ok,
                   "PASS" if elig_ok else "FAIL"))

    # 4. Classifications
    class_refs = [(.7,"Supported"),(-.3,"Contradicted"),(0,"Undetermined"),
                  (.2,"Supported"),(-.2,"Contradicted"),(.19,"Undetermined"),
                  (-.19,"Undetermined"),(None,"Abstain")]
    class_ok = all(classify(v) == e for v, e in class_refs)
    checks.append(("classification boundaries", class_ok,
                   "PASS" if class_ok else "FAIL"))

    # 5. Aggregate outputs
    agg_ok = all(close(aggregate(CASES[c], m), EXPECTED[c][m])
                 for c in EXPECTED for m in MODELS)
    checks.append(("aggregate outputs A-I", agg_ok,
                   "PASS" if agg_ok else "FAIL"))

    # 6. Removal bound — case G
    sens = removal_sensitivity(CASES['G'], 1)
    sens_ok = all(r['delta'] is not None and r['bound'] is not None
                  and r['delta'] <= r['bound'] + 1e-12 for r in sens.values())
    checks.append(("removal bound for case G", sens_ok,
                   "PASS" if sens_ok else "FAIL"))

    # 7. Model-specific removal accounting — case I
    si = removal_sensitivity(CASES['I'], 1)
    si_ok = (si["M1'"]["delta"] == 0 and si["M1'"]["bound"] == 0
             and si['M2']["delta"] == 0 and si['M2']["bound"] == 0)
    checks.append(("model-specific removal accounting", si_ok,
                   "PASS" if si_ok else "FAIL"))

    # 8. NEW: Removal sensitivity across all cases
    all_rem_ok, all_rem_msg = check_removal_all_cases()
    checks.append(("removal sensitivity across all cases", all_rem_ok, all_rem_msg))

    # 9. NEW: High-precision raw weight check
    exact_ok, exact_msg = check_raw_weights_exact()
    checks.append(("exact raw weights (tolerance 1e-12)", exact_ok, exact_msg))

    # 10. NEW: List acceptance
    list_ok, list_msg = check_list_acceptance()
    checks.append(("list Q accepted", list_ok, list_msg))

    # Print aggregate outputs
    print("AGGREGATE OUTPUTS")
    print(f"{'Case':<8}" + ''.join(f"{m:>10}" for m in MODELS))
    for c, items in CASES.items():
        print(f"{c:<8}" + ''.join(f"{fmt(aggregate(items, m)):>10}" for m in MODELS))

    # Print removal sensitivity for G
    print("\nREMOVAL SENSITIVITY — G, remove e2")
    for m, r in sens.items():
        print(f"{m:4} before={fmt(r['before'], 6)} "
              f"after={fmt(r['after'], 6)} "
              f"delta={fmt(r['delta'], 6)} "
              f"bound={fmt(r['bound'], 6)}")

    # Print check summary
    print("\nCHECK SUMMARY")
    for name, ok, msg in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok and msg:
            print(f"      {msg}")

    passed = all(ok for _, ok, _ in checks)
    print("OVERALL:", "ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED")
    return passed


if __name__ == '__main__':
    raise SystemExit(0 if run_checks() else 1)
