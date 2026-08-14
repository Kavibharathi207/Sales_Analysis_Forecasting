"""
test_all.py
===========
Standalone tests for all 3 backend services.
Runs directly — no server required.

Usage:
    cd c:\\Users\\kavib\\Desktop\\Sales_Analysis_Forecasting
    python test_all.py
"""

import sys
import os

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.schemas.anomaly import AnomalyDetectRequest, ForecastPoint as AF, ActualPoint
from backend.schemas.whatif import WhatIfRequest, ForecastPoint as WF
from backend.schemas.recommendation import RecommendationRequest, ForecastPoint as RF, AnomalySummary
from backend.services.anomaly_service import detect_anomalies
from backend.services.whatif_service import run_what_if
from backend.services.recommendation_service import generate_recommendations

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# TEST 1 — ANOMALY DETECTION
# Scenario: Actual=2500, Forecast=1080 → should be ANOMALY HIGH
# ─────────────────────────────────────────────────────────────
separator("TEST 1: ANOMALY DETECTION")

req = AnomalyDetectRequest(
    product="Paracetamol",
    forecast=[
        AF(date="2026-08-15", predicted_sales=1000),
        AF(date="2026-08-16", predicted_sales=1050),
        AF(date="2026-08-17", predicted_sales=1100),
        AF(date="2026-08-18", predicted_sales=1080),   # ← anomaly day
        AF(date="2026-08-19", predicted_sales=1200),
    ],
    actuals=[
        ActualPoint(date="2026-08-15", actual_sales=1020),
        ActualPoint(date="2026-08-16", actual_sales=1070),
        ActualPoint(date="2026-08-17", actual_sales=1090),
        ActualPoint(date="2026-08-18", actual_sales=2500),  # ← spike
        ActualPoint(date="2026-08-19", actual_sales=1180),
    ],
)

resp = detect_anomalies(req)

print(f"\nProduct      : {resp.product}")
print(f"Total days   : {resp.total_days}")
print(f"Anomaly count: {resp.anomaly_count}")
print(f"\n{'Date':<14} {'Actual':>8} {'Forecast':>10} {'Deviation%':>12} {'Status':<12} {'Severity'}")
print("-" * 65)
for r in resp.results:
    print(f"{r.date:<14} {r.actual_sales:>8.1f} {r.forecast_sales:>10.1f} {r.deviation_percent:>11.2f}% {r.status:<12} {r.severity}")

# Validate the key case: 2026-08-18 actual=2500, forecast=1080
aug18 = next(r for r in resp.results if r.date == "2026-08-18")
expected_dev = round(abs(2500 - 1080) / 1080 * 100, 2)  # 131.48%

t1a = aug18.status == "anomaly"
t1b = aug18.severity == "high"
t1c = abs(aug18.deviation_percent - expected_dev) < 0.1

print(f"\n[Check] 2026-08-18 deviation = {aug18.deviation_percent}% (expected ~{expected_dev}%) → {PASS if t1c else FAIL}")
print(f"[Check] status = '{aug18.status}' (expected 'anomaly')                    → {PASS if t1a else FAIL}")
print(f"[Check] severity = '{aug18.severity}' (expected 'high')                   → {PASS if t1b else FAIL}")


# ─────────────────────────────────────────────────────────────
# TEST 2 — WHAT-IF ANALYSIS
# Scenario: Forecast=10000 total, +20% → should give 12000
# ─────────────────────────────────────────────────────────────
separator("TEST 2: WHAT-IF ANALYSIS")

req2 = WhatIfRequest(
    product="Paracetamol",
    forecast=[
        WF(date="2026-08-15", predicted_sales=2000),
        WF(date="2026-08-16", predicted_sales=2000),
        WF(date="2026-08-17", predicted_sales=2000),
        WF(date="2026-08-18", predicted_sales=2000),
        WF(date="2026-08-19", predicted_sales=2000),
    ],
    change_percent=20.0,
)

resp2 = run_what_if(req2)

print(f"\nProduct          : {resp2.product}")
print(f"Change applied   : +{resp2.change_percent}%")
print(f"Total baseline   : {resp2.total_baseline}")
print(f"Total adjusted   : {resp2.total_adjusted}")
print(f"Total difference : +{resp2.total_difference}")
print(f"\n{'Date':<14} {'Baseline':>10} {'Adjusted':>10} {'Diff':>8} {'Chg%':>8}")
print("-" * 55)
for r in resp2.results:
    print(f"{r.date:<14} {r.baseline_sales:>10.1f} {r.adjusted_sales:>10.1f} {r.difference:>8.1f} {r.change_percent:>7.1f}%")

t2a = resp2.total_baseline == 10000.0
t2b = resp2.total_adjusted == 12000.0
t2c = resp2.total_difference == 2000.0

print(f"\n[Check] total_baseline = {resp2.total_baseline} (expected 10000.0)  → {PASS if t2a else FAIL}")
print(f"[Check] total_adjusted = {resp2.total_adjusted} (expected 12000.0)  → {PASS if t2b else FAIL}")
print(f"[Check] total_difference = {resp2.total_difference} (expected 2000.0) → {PASS if t2c else FAIL}")


# ─────────────────────────────────────────────────────────────
# TEST 2b — WHAT-IF WITH SUPPLY DISRUPTION
# Scenario: +20% but 2 days zeroed out
# ─────────────────────────────────────────────────────────────
separator("TEST 2b: WHAT-IF + SUPPLY DISRUPTION")

req2b = WhatIfRequest(
    product="Paracetamol",
    forecast=[
        WF(date="2026-08-15", predicted_sales=2000),
        WF(date="2026-08-16", predicted_sales=2000),
        WF(date="2026-08-17", predicted_sales=2000),
        WF(date="2026-08-18", predicted_sales=2000),
        WF(date="2026-08-19", predicted_sales=2000),
    ],
    change_percent=20.0,
    disruption_start="2026-08-17",
    disruption_end="2026-08-18",
)

resp2b = run_what_if(req2b)

print(f"\nDisruption period: 2026-08-17 → 2026-08-18")
print(f"Disruption days  : {resp2b.disruption_days}")
print(f"Total adjusted   : {resp2b.total_adjusted} (3 normal days × 2400 = 7200)")
print(f"\n{'Date':<14} {'Baseline':>10} {'Adjusted':>10} {'Note'}")
print("-" * 50)
for r in resp2b.results:
    note = "← ZEROED (disruption)" if r.adjusted_sales == 0 else ""
    print(f"{r.date:<14} {r.baseline_sales:>10.1f} {r.adjusted_sales:>10.1f}  {note}")

t2d = resp2b.disruption_days == 2
t2e = resp2b.total_adjusted == 7200.0

print(f"\n[Check] disruption_days = {resp2b.disruption_days} (expected 2)       → {PASS if t2d else FAIL}")
print(f"[Check] total_adjusted = {resp2b.total_adjusted} (expected 7200.0) → {PASS if t2e else FAIL}")


# ─────────────────────────────────────────────────────────────
# TEST 3 — RECOMMENDATION ENGINE
# Scenario: Forecast growing +20% → RESTOCK_ALERT HIGH
# ─────────────────────────────────────────────────────────────
separator("TEST 3: RECOMMENDATION ENGINE — GROWTH SCENARIO")

req3 = RecommendationRequest(
    product="Paracetamol",
    forecast=[
        RF(date="2026-08-15", predicted_sales=1000),
        RF(date="2026-08-16", predicted_sales=1050),
        RF(date="2026-08-17", predicted_sales=1100),
        RF(date="2026-08-18", predicted_sales=1150),
        RF(date="2026-08-19", predicted_sales=1200),
        RF(date="2026-08-20", predicted_sales=1250),
        RF(date="2026-08-21", predicted_sales=1300),
        RF(date="2026-08-22", predicted_sales=1350),
        RF(date="2026-08-23", predicted_sales=1400),
        RF(date="2026-08-24", predicted_sales=1450),
    ],
    anomaly_summary=AnomalySummary(anomaly_count=1, high_severity_count=1),
    model_mape=60.75,  # real M01AB prophet MAPE from evaluation CSV
)

resp3 = generate_recommendations(req3)

print(f"\nProduct          : {resp3.product}")
print(f"Forecast trend   : +{resp3.forecast_trend_pct}%")
print(f"\nRecommendations ({len(resp3.recommendations)} total):")
for rec in resp3.recommendations:
    print(f"\n  [{rec.priority}] {rec.signal}")
    print(f"  → {rec.recommendation}")
    print(f"  Rationale: {rec.rationale}")

signals = [r.signal for r in resp3.recommendations]
priorities = [r.priority for r in resp3.recommendations]

t3a = "RESTOCK_ALERT" in signals
t3b = "DEMAND_SPIKE" in signals
t3c = "HIGH_UNCERTAINTY" in signals
t3d = priorities[0] == "HIGH"  # highest priority first

print(f"\n[Check] RESTOCK_ALERT triggered (trend +{resp3.forecast_trend_pct}%)  → {PASS if t3a else FAIL}")
print(f"[Check] DEMAND_SPIKE triggered (1 high-severity anomaly)  → {PASS if t3b else FAIL}")
print(f"[Check] HIGH_UNCERTAINTY triggered (MAPE=60.75%)          → {PASS if t3c else FAIL}")
print(f"[Check] First recommendation is HIGH priority              → {PASS if t3d else FAIL}")


# ─────────────────────────────────────────────────────────────
# TEST 3b — RECOMMENDATION: DECLINE SCENARIO
# ─────────────────────────────────────────────────────────────
separator("TEST 3b: RECOMMENDATION ENGINE — DECLINE SCENARIO")

req3b = RecommendationRequest(
    product="N02BE",
    forecast=[
        RF(date="2026-08-15", predicted_sales=500),
        RF(date="2026-08-16", predicted_sales=480),
        RF(date="2026-08-17", predicted_sales=460),
        RF(date="2026-08-18", predicted_sales=440),
        RF(date="2026-08-19", predicted_sales=420),
        RF(date="2026-08-20", predicted_sales=400),
        RF(date="2026-08-21", predicted_sales=380),
        RF(date="2026-08-22", predicted_sales=360),
        RF(date="2026-08-23", predicted_sales=340),
        RF(date="2026-08-24", predicted_sales=320),
    ],
)

resp3b = generate_recommendations(req3b)

print(f"\nProduct          : {resp3b.product}")
print(f"Forecast trend   : {resp3b.forecast_trend_pct}%")
for rec in resp3b.recommendations:
    print(f"\n  [{rec.priority}] {rec.signal}")
    print(f"  → {rec.recommendation}")

t3e = any(r.signal == "OVERSTOCK_RISK" for r in resp3b.recommendations)
print(f"\n[Check] OVERSTOCK_RISK triggered (trend {resp3b.forecast_trend_pct}%) → {PASS if t3e else FAIL}")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
separator("TEST SUMMARY")

all_tests = [t1a, t1b, t1c, t2a, t2b, t2c, t2d, t2e, t3a, t3b, t3c, t3d, t3e]
passed = sum(all_tests)
total = len(all_tests)

print(f"\n  Passed : {passed} / {total}")
print(f"  Status : {'ALL TESTS PASSED ✅' if passed == total else f'{total - passed} TESTS FAILED ❌'}")
print()
