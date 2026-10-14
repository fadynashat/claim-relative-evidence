#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible controlled cases A-I for the EWDA/P1 evidence model.

Implements the four models in the manuscript appendix and checks numerical
references, boundaries, malformed inputs, classifications, and deletion bounds.
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


def valid_number(x: Any, lo: float, hi: float) -> bool:
    return (not isinstance(x, bool) and isinstance(x, (int, float))
            and math.isfinite(x) and lo <= x <= hi)


def valid_q(q: Any) -> bool:
    return isinstance(q, tuple) and len(q) == K and all(valid_number(x, 0.0, 1.0) for x in q)


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


def run_checks() -> bool:
    checks: list[tuple[str,bool]] = []
    checks.append(("valid cases and malformed input", all(validate_inputs(v) for v in CASES.values()) and all(not validate_inputs(x) for x in [None, {}, [None], [{}], [{"Q":(1,2),"s":0,"status":"ACCEPTED"}], [{"Q":(1,1,True),"s":0,"status":"ACCEPTED"}], [{"Q":(1,1,float('nan')),"s":0,"status":"ACCEPTED"}], [{"Q":(1,1,float('inf')),"s":0,"status":"ACCEPTED"}], [{"Q":(1,1,1),"s":True,"status":"ACCEPTED"}], [{"Q":(1,1,1),"s":0,"status":None}], [{"Q":(1,1,1),"status":"ACCEPTED"}]])))
    weight_refs=[((.9,.9,.8),.8653497422),((.8,.8,.9),.8320335292),((.2,.9,.8),.5241482788),((.9,.25,.9),.5872301462),((.1,.1,.1),.1),((.31,.9,.8),.6065939346),((.29,.9,.8),.5932578563),((.9,.9,.3),.6240251469)]
    weight_ok=all(abs(raw_weight(q)-e)<1e-9 for q,e in weight_refs)
    for q in [(),(.2,.3),(.2,.3,1.2),(.2,.3,float('nan'))]:
        try: raw_weight(q); weight_ok=False
        except ValueError: pass
    checks.append(("raw weights and domain guards",weight_ok))
    elig=[((.9,.9,.8),1),((.2,.9,.8),0),((.9,.25,.9),0),((.1,.1,.1),0),((.31,.9,.8),1),((.29,.9,.8),0),((.9,.9,.3),1),((0,.9,.9),0),((.30,.9,.8),1),((.9,.30,.8),1),((.9,.9,.40),1),((.35,.35,.35),0),((.40,.30,.30),1)]
    checks.append(("eligibility including boundaries",all(numerical_eligibility(q)==e for q,e in elig)))
    class_refs=[(.7,"Supported"),(-.3,"Contradicted"),(0,"Undetermined"),(.2,"Supported"),(-.2,"Contradicted"),(.19,"Undetermined"),(-.19,"Undetermined"),(None,"Abstain")]
    checks.append(("classification boundaries",all(classify(v)==e for v,e in class_refs)))
    checks.append(("aggregate outputs A-I",all(close(aggregate(CASES[c],m),EXPECTED[c][m]) for c in EXPECTED for m in MODELS)))
    sens=removal_sensitivity(CASES['G'],1)
    checks.append(("removal bound for case G",all(r['delta'] is not None and r['bound'] is not None and r['delta'] <= r['bound']+1e-12 for r in sens.values())))
    si=removal_sensitivity(CASES['I'],1)
    checks.append(("model-specific removal accounting",si["M1'"]["delta"]==0 and si["M1'"]["bound"]==0 and si['M2']["delta"]==0 and si['M2']["bound"]==0))
    print("AGGREGATE OUTPUTS")
    print(f"{'Case':<8}"+''.join(f"{m:>10}" for m in MODELS))
    for c,items in CASES.items(): print(f"{c:<8}"+''.join(f"{fmt(aggregate(items,m)):>10}" for m in MODELS))
    print("\nREMOVAL SENSITIVITY — G, remove e2")
    for m,r in sens.items(): print(f"{m:4} before={fmt(r['before'],6)} after={fmt(r['after'],6)} delta={fmt(r['delta'],6)} bound={fmt(r['bound'],6)}")
    print("\nCHECK SUMMARY")
    for name,ok in checks: print(f"{'PASS' if ok else 'FAIL'}  {name}")
    passed=all(ok for _,ok in checks)
    print("OVERALL:","ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED")
    return passed

if __name__ == '__main__':
    raise SystemExit(0 if run_checks() else 1)
