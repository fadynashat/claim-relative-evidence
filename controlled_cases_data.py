
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled test cases A-I for:
"Claim-Relative Evidence Quality and Support in AI-Augmented Legacy Modernization"

This script reproduces every numeric value reported in Appendix A of the paper
and runs independent sanity checks on:
  - input validation (including NaN and Infinity)
  - raw weights
  - numerical eligibility
  - aggregate outputs
  - classifications
  - removal-sensitivity bounds
  - boundary cases

Every check function returns a Boolean. The runner aggregates all results
into a single OVERALL verdict and exits with a non-zero status on failure.

Run with: python3 controlled_cases_data.py
"""

import math
from typing import List, Tuple, Dict, Optional, Callable


# ============================================================
# 0. Explicit type aliases
# ============================================================

AggregateResult = Optional[float]
Classification = str  # Supported | Contradicted | Undetermined | Abstain


# ============================================================
# 1. Fixed configuration (matches Appendix A.1)
# ============================================================

K = 3
CRITICAL_DIMENSIONS = [0, 1]
TAU_C = 0.30
TAU_W = 0.40
THETA_MINUS = -0.20
THETA_PLUS = 0.20


# ============================================================
# 2. Core functions
# ============================================================

def raw_weight(q: Tuple[float, ...]) -> float:
    """Raw geometric-mean weight, Definition 4.2."""
    product = 1.0
    for qi in q:
        product *= qi
    return product ** (1.0 / len(q))


def numerical_eligibility(q: Tuple[float, ...]) -> int:
    """Numerical eligibility indicator A(e,c), Definition 4.3."""
    for i in CRITICAL_DIMENSIONS:
        if q[i] < TAU_C:
            return 0
    if max(q) < TAU_W:
        return 0
    if raw_weight(q) <= 0.0:
        return 0
    return 1


def validate_inputs(items: List[Dict]) -> bool:
    """
    Validate that every evidence item conforms to the model's domain.

    Rejects:
      - tuples of wrong length
      - non-numeric values (including bool)
      - NaN or +/-Infinity
      - Q values outside [0, 1]
      - s values outside [-1, 1]
      - unknown status strings
    """
    for idx, it in enumerate(items):
        q = it.get("Q")
        s = it.get("s")
        status = it.get("status")

        # --- Q validation ---
        if not isinstance(q, tuple) or len(q) != K:
            print(f"  INPUT ERROR: item {idx} has invalid Q shape: {q}")
            return False
        for qi in q:
            if (isinstance(qi, bool)
                    or not isinstance(qi, (int, float))
                    or not math.isfinite(qi)
                    or not (0.0 <= qi <= 1.0)):
                print(f"  INPUT ERROR: item {idx} has invalid Q: {q}")
                return False

        # --- s validation ---
        if (isinstance(s, bool)
                or not isinstance(s, (int, float))
                or not math.isfinite(s)
                or not (-1.0 <= s <= 1.0)):
            print(f"  INPUT ERROR: item {idx} has invalid s: {s}")
            return False

        # --- status validation ---
        if status not in ("ACCEPTED", "REJECTED", "PENDING"):
            print(f"  INPUT ERROR: item {idx} has invalid status: {status}")
            return False

    return True


def aggregate(items: List[Dict], model: str) -> AggregateResult:
    """Compute S_M(c). Returns float, or None for the undefined state (⊥)."""
    numerator = 0.0
    denominator = 0.0

    for it in items:
        q = it["Q"]
        s = it["s"]
        status = it["status"]

        if model == "M0":
            w = 1.0
        elif model == "M1":
            w = raw_weight(q)
        elif model == "M1'":
            if status != "ACCEPTED":
                continue
            w = raw_weight(q)
        elif model == "M2":
            if status != "ACCEPTED":
                continue
            if numerical_eligibility(q) != 1:
                continue
            w = raw_weight(q)
        else:
            raise ValueError(f"Unknown model: {model}")

        numerator += w * s
        denominator += w

    if denominator == 0.0:
        return None
    return numerator / denominator


def classify(value: AggregateResult) -> Classification:
    """Map aggregate to Supported / Contradicted / Undetermined / Abstain (⊥)."""
    if value is None:
        return "Abstain"
    if value >= THETA_PLUS:
        return "Supported"
    if value <= THETA_MINUS:
        return "Contradicted"
    return "Undetermined"


def fmt(value: AggregateResult, decimals: int = 4) -> str:
    if value is None:
        return "⊥"
    return f"{value:.{decimals}f}"


# ============================================================
# 3. Case definitions (matches Appendix A.2)
# ============================================================

CASES: Dict[str, List[Dict]] = {
    "A": [
        {"Q": (0.9, 0.9, 0.8), "s": +0.8, "status": "ACCEPTED"},
        {"Q": (0.8, 0.8, 0.9), "s": +0.6, "status": "ACCEPTED"},
    ],
    "B": [
        {"Q": (0.9, 0.9, 0.8), "s": +0.8, "status": "ACCEPTED"},
        {"Q": (0.2, 0.9, 0.8), "s": -0.9, "status": "ACCEPTED"},
    ],
    "C": [
        {"Q": (0.9, 0.9, 0.8), "s": +0.8, "status": "ACCEPTED"},
        {"Q": (0.9, 0.25, 0.9), "s": +0.9, "status": "ACCEPTED"},
    ],
    "D": [
        {"Q": (0.9, 0.9, 0.8), "s": +0.8, "status": "ACCEPTED"},
        {"Q": (0.8, 0.8, 0.9), "s": -0.8, "status": "ACCEPTED"},
    ],
    "E": [
        {"Q": (0.2, 0.9, 0.8), "s": +0.8, "status": "ACCEPTED"},
        {"Q": (0.1, 0.1, 0.1), "s": -0.9, "status": "ACCEPTED"},
    ],
    "F": [
        {"Q": (0.31, 0.9, 0.8), "s": +0.8, "status": "ACCEPTED"},
        {"Q": (0.29, 0.9, 0.8), "s": +0.7, "status": "ACCEPTED"},
    ],
    "G": [
        {"Q": (0.9, 0.9, 0.8), "s": +0.8, "status": "ACCEPTED"},
        {"Q": (0.8, 0.8, 0.9), "s": +0.6, "status": "ACCEPTED"},
    ],
    "H": [
        {"Q": (0.9, 0.9, 0.8), "s": +0.8, "status": "ACCEPTED"},
        {"Q": (0.9, 0.9, 0.3), "s": +0.7, "status": "ACCEPTED"},
    ],
    "I": [
        {"Q": (0.9, 0.9, 0.8), "s": +0.8, "status": "ACCEPTED"},
        {"Q": (0.8, 0.8, 0.9), "s": +0.6, "status": "REJECTED"},
    ],
}

MODELS = ["M0", "M1", "M1'", "M2"]


# ============================================================
# 4. Removal-sensitivity computation for case G
# ============================================================

def removal_sensitivity_G() -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}

    for model in MODELS:
        items = CASES["G"]
        s_before = aggregate(items, model)
        s_after = aggregate([items[0]], model)

        if model == "M0":
            w_removed = 1.0
            W_before = float(len(items))
        else:
            w_removed = raw_weight(items[1]["Q"])
            W_before = sum(raw_weight(it["Q"]) for it in items)

        if s_before is None or s_after is None:
            results[model] = {
                "before": s_before, "after": s_after,
                "actual_change": None, "bound": None,
            }
            continue

        results[model] = {
            "before": s_before,
            "after": s_after,
            "actual_change": abs(s_before - s_after),
            "bound": 2.0 * w_removed / W_before,
        }
    return results


# ============================================================
# 5. Reporting tables
# ============================================================

def print_raw_weights_and_eligibility() -> None:
    print("\n" + "=" * 78)
    print("TABLE A.2 — Raw weights and numerical eligibility")
    print("=" * 78)
    print(f"{'Case':<6}{'Evidence':<10}{'w_raw':>16}{'A':>6}")
    print("-" * 78)
    for case_name, items in CASES.items():
        for idx, it in enumerate(items):
            w = raw_weight(it["Q"])
            a = numerical_eligibility(it["Q"])
            print(f"{case_name:<6}{'e'+str(idx+1):<10}{w:>16.10f}{a:>6}")
    print("=" * 78)


def print_aggregates() -> None:
    print("\n" + "=" * 78)
    print("TABLE A.3 — Computed outputs for cases A-I")
    print("=" * 78)
    print(f"{'Case':<8}" + "".join(f"{m:>14}" for m in MODELS))
    print("-" * 78)
    for case_name, items in CASES.items():
        row = f"{case_name:<8}"
        for model in MODELS:
            row += f"{fmt(aggregate(items, model)):>14}"
        print(row)
    print("=" * 78)


def print_classifications() -> None:
    print("\n" + "=" * 78)
    print("TABLE A.4 — Classification thresholds: "
          "theta_minus = -0.20, theta_plus = 0.20 (inclusive)")
    print("=" * 78)
    print(f"{'Case':<8}" + "".join(f"{m:>16}" for m in MODELS))
    print("-" * 78)
    for case_name, items in CASES.items():
        row = f"{case_name:<8}"
        for model in MODELS:
            row += f"{classify(aggregate(items, model)):>16}"
        print(row)
    print("=" * 78)


def print_removal_case_G() -> None:
    print("\n" + "=" * 78)
    print("TABLE A.5 — Removal-sensitivity for case G (remove e_2)")
    print("=" * 78)
    print(f"{'Model':<8}{'S_before':>12}{'S_after':>12}"
          f"{'|Δ|':>12}{'Bound':>14}")
    print("-" * 78)
    for model, r in removal_sensitivity_G().items():
        if r["before"] is None or r["after"] is None:
            print(f"{model:<8}{'⊥':>12}{'⊥':>12}{'—':>12}{'—':>14}")
        else:
            print(f"{model:<8}{r['before']:>12.6f}{r['after']:>12.6f}"
                  f"{r['actual_change']:>12.6f}{r['bound']:>14.6f}")
    print("=" * 78)


# ============================================================
# 6. Independent sanity checks — all return Boolean
# ============================================================

def check_input_validation() -> bool:
    """Verify that valid inputs pass and invalid inputs are rejected."""
    print("\n" + "-" * 78)
    print("CHECK 0 — Input validation")
    print("-" * 78)

    all_ok = True

    # Valid inputs
    for case_name, items in CASES.items():
        if not validate_inputs(items):
            print(f"  FAIL: case {case_name} rejected as invalid")
            all_ok = False
    if all_ok:
        print("  PASS: all 9 cases accepted as valid inputs")

    # Invalid inputs
    bad_cases = [
        ([{"Q": (0.9, 0.9), "s": 0.5, "status": "ACCEPTED"}], "wrong Q length"),
        ([{"Q": (0.9, 0.9, 1.5), "s": 0.5, "status": "ACCEPTED"}], "Q > 1"),
        ([{"Q": (0.9, 0.9, -0.1), "s": 0.5, "status": "ACCEPTED"}], "Q < 0"),
        ([{"Q": (0.9, 0.9, 0.8), "s": 1.5, "status": "ACCEPTED"}], "s > 1"),
        ([{"Q": (0.9, 0.9, 0.8), "s": 0.5, "status": "UNKNOWN"}], "bad status"),
        ([{"Q": (0.9, 0.9, 0.8), "s": True, "status": "ACCEPTED"}], "s is bool"),
        ([{"Q": (0.9, 0.9, True), "s": 0.5, "status": "ACCEPTED"}], "Q contains bool"),
        ([{"Q": (0.9, 0.9, "0.8"), "s": 0.5, "status": "ACCEPTED"}], "Q contains str"),
        ([{"Q": (0.9, 0.9, 0.8), "s": None, "status": "ACCEPTED"}], "s is None"),
        ([{"Q": (0.9, 0.9, float("nan")), "s": 0.5, "status": "ACCEPTED"}], "Q contains NaN"),
        ([{"Q": (0.9, 0.9, float("inf")), "s": 0.5, "status": "ACCEPTED"}], "Q contains +Inf"),
        ([{"Q": (0.9, 0.9, float("-inf")), "s": 0.5, "status": "ACCEPTED"}], "Q contains -Inf"),
        ([{"Q": (0.9, 0.9, 0.8), "s": float("nan"), "status": "ACCEPTED"}], "s is NaN"),
        ([{"Q": (0.9, 0.9, 0.8), "s": float("inf"), "status": "ACCEPTED"}], "s is +Inf"),
        ([{"Q": (0.9, 0.9, 0.8), "s": float("-inf"), "status": "ACCEPTED"}], "s is -Inf"),
    ]
    for items, label in bad_cases:
        if validate_inputs(items):
            print(f"  FAIL: invalid input accepted ({label})")
            all_ok = False
        else:
            print(f"  PASS: invalid input rejected ({label})")

    return all_ok


def check_raw_weights() -> bool:
    """
    Verify raw weights against independent reference values.

    Note on independence: the reference values below were computed
    independently using high-precision arithmetic before being hard-coded
    here. They are not derived from `raw_weight()`; the comparison tests
    whether the implementation matches the mathematical definition.
    """
    print("\n" + "-" * 78)
    print("CHECK 1 — Raw weights (independent reference values)")
    print("-" * 78)
    references = [
        ((0.9, 0.9, 0.8), 0.8653497422),
        ((0.8, 0.8, 0.9), 0.8320335292),
        ((0.2, 0.9, 0.8), 0.5241482788),
        ((0.9, 0.25, 0.9), 0.5872301462),
        ((0.1, 0.1, 0.1), 0.1000000000),
        ((0.31, 0.9, 0.8), 0.6065939346),
        ((0.29, 0.9, 0.8), 0.5932578563),
        ((0.9, 0.9, 0.3), 0.6240251469),
    ]
    ok = True
    for q, expected in references:
        got = raw_weight(q)
        passed = abs(got - expected) < 1e-9
        status = "PASS" if passed else "FAIL"
        if not passed:
            ok = False
        print(f"  {q} → {got:.10f}  expected {expected:.10f}  [{status}]")
    return ok


def check_numerical_eligibility() -> bool:
    """Verify numerical eligibility, including boundary values."""
    print("\n" + "-" * 78)
    print("CHECK 2 — Numerical eligibility (including boundaries)")
    print("-" * 78)
    references = [
        ((0.9, 0.9, 0.8), 1, "well above thresholds"),
        ((0.8, 0.8, 0.9), 1, "well above thresholds"),
        ((0.2, 0.9, 0.8), 0, "q_1 below tau_c"),
        ((0.9, 0.25, 0.9), 0, "q_2 below tau_c"),
        ((0.1, 0.1, 0.1), 0, "all below tau_w"),
        ((0.31, 0.9, 0.8), 1, "just above tau_c"),
        ((0.29, 0.9, 0.8), 0, "just below tau_c"),
        ((0.9, 0.9, 0.3), 1, "non-critical dim low"),
        ((0.0, 0.9, 0.9), 0, "zero coordinate"),
        ((0.30, 0.9, 0.8), 1, "critical exactly at tau_c"),
        ((0.9, 0.30, 0.8), 1, "second critical at tau_c"),
        ((0.9, 0.9, 0.40), 1, "max exactly at tau_w"),
        ((0.35, 0.35, 0.35), 0, "max below tau_w"),
        ((0.40, 0.30, 0.30), 1, "max exactly at tau_w, critical at tau_c"),
    ]
    ok = True
    for q, expected, label in references:
        got = numerical_eligibility(q)
        passed = (got == expected)
        status = "PASS" if passed else "FAIL"
        if not passed:
            ok = False
        print(f"  {str(q):<22} A={got}  expected {expected}  [{status}]  ({label})")
    return ok


def check_classifications() -> bool:
    """Verify classification outcomes, including inclusive boundaries."""
    print("\n" + "-" * 78)
    print("CHECK 3 — Classification outcomes (inclusive boundaries)")
    print("-" * 78)
    references = [
        (0.70, "Supported"),
        (-0.30, "Contradicted"),
        (0.00, "Undetermined"),
        (0.20, "Supported"),
        (-0.20, "Contradicted"),
        (0.19, "Undetermined"),
        (-0.19, "Undetermined"),
        (None, "Abstain"),
    ]
    ok = True
    for value, expected in references:
        got = classify(value)
        passed = (got == expected)
        status = "PASS" if passed else "FAIL"
        if not passed:
            ok = False
        display = "⊥" if value is None else f"{value:+.2f}"
        print(f"  S={display:>6} → {got:<14} expected {expected:<14} [{status}]")
    return ok


def check_aggregate_outputs() -> bool:
    """Verify aggregate outputs against published values."""
    print("\n" + "-" * 78)
    print("CHECK 4 — Aggregate outputs (Appendix A.3)")
    print("-" * 78)
    expected = {
        "A":  {"M0": 0.7000, "M1": 0.7020, "M1'": 0.7020, "M2": 0.7020},
        "B":  {"M0": -0.0500, "M1": 0.1587, "M1'": 0.1587, "M2": 0.8000},
        "C":  {"M0": 0.8500, "M1": 0.8404, "M1'": 0.8404, "M2": 0.8000},
        "D":  {"M0": 0.0000, "M1": 0.0157, "M1'": 0.0157, "M2": 0.0157},
        "E":  {"M0": -0.0500, "M1": 0.5276, "M1'": 0.5276, "M2": None},
        "F":  {"M0": 0.7500, "M1": 0.7506, "M1'": 0.7506, "M2": 0.8000},
        "G":  {"M0": 0.7000, "M1": 0.7020, "M1'": 0.7020, "M2": 0.7020},
        "H":  {"M0": 0.7500, "M1": 0.7581, "M1'": 0.7581, "M2": 0.7581},
        "I":  {"M0": 0.7000, "M1": 0.7020, "M1'": 0.8000, "M2": 0.8000},
    }
    tolerance = 5e-4
    ok = True
    for case_name, exp in expected.items():
        for model, e in exp.items():
            got = aggregate(CASES[case_name], model)
            if e is None:
                passed = (got is None)
            else:
                passed = (got is not None) and (abs(got - e) < tolerance)
            status = "PASS" if passed else "FAIL"
            if not passed:
                ok = False
            print(f"  Case {case_name:>2} | {model:<3} | "
                  f"expected {str(e):>7} | got {fmt(got):>7} | {status}")
    return ok


def check_removal_bound() -> bool:
    """Verify that removal-sensitivity bounds hold in case G."""
    print("\n" + "-" * 78)
    print("CHECK 5 — Removal-sensitivity bound (case G)")
    print("-" * 78)
    ok = True
    for model, r in removal_sensitivity_G().items():
        if r["actual_change"] is None or r["bound"] is None:
            print(f"  FAIL: {model}: removal bound is undefined")
            ok = False
            continue
        passed = r["actual_change"] <= r["bound"] + 1e-12
        status = "PASS" if passed else "FAIL"
        if not passed:
            ok = False
        print(f"  {model}: |Δ|={r['actual_change']:.6f} ≤ "
              f"bound={r['bound']:.6f}  [{status}]")
    return ok


# ============================================================
# 7. Runner that aggregates all check results
# ============================================================

def run_all_checks() -> bool:
    checks: List[Tuple[str, Callable[[], bool]]] = [
        ("Input validation", check_input_validation),
        ("Raw weights", check_raw_weights),
        ("Numerical eligibility", check_numerical_eligibility),
        ("Classifications", check_classifications),
        ("Aggregate outputs", check_aggregate_outputs),
        ("Removal bound", check_removal_bound),
    ]
    print("\n" + "=" * 78)
    print("RUNNING ALL CHECKS")
    print("=" * 78)
    results: List[bool] = []
    for name, check in checks:
        try:
            result = check()
            if result is None:
                print(f"WARNING: {name} did not return a boolean.")
                results.append(False)
            else:
                results.append(result)
        except Exception as exc:
            print(f"ERROR: {name}: {exc}")
            results.append(False)

    print("\n" + "=" * 78)
    print("CHECK SUMMARY")
    print("=" * 78)
    for (name, _), passed in zip(checks, results):
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<28} [{status}]")
    return all(results)


# ============================================================
# 8. Entry point
# ============================================================

if __name__ == "__main__":
    # Guard: refuse to run on invalid data
    if not all(validate_inputs(items) for items in CASES.values()):
        print("\nABORT: invalid input detected. Fix the data before running.")
        raise SystemExit(1)

    print_raw_weights_and_eligibility()
    print_aggregates()
    print_classifications()
    print_removal_case_G()

    all_pass = run_all_checks()

    print("\n" + "=" * 78)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 78)

    raise SystemExit(0 if all_pass else 1)
