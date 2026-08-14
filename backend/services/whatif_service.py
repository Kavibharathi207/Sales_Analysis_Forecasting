"""
whatif_service.py
=================
Business logic for what-if scenario analysis.

Applies a percentage change to each forecast point.
Optionally zeroes out a supply disruption date range.
"""

from typing import List
from backend.schemas.whatif import (
    WhatIfRequest,
    WhatIfResponse,
    WhatIfPoint,
)


def run_what_if(request: WhatIfRequest) -> WhatIfResponse:
    factor = 1.0 + request.change_percent / 100.0
    disruption_days = 0
    results: List[WhatIfPoint] = []

    for fp in request.forecast:
        baseline = fp.predicted_sales
        adjusted = baseline * factor

        # Supply disruption: zero out the range
        if request.disruption_start and request.disruption_end:
            if request.disruption_start <= fp.date <= request.disruption_end:
                adjusted = 0.0
                disruption_days += 1

        adjusted = max(adjusted, 0.0)  # floor at 0
        difference = round(adjusted - baseline, 4)

        # Per-row change_percent (handles disruption zeros cleanly)
        if baseline != 0:
            row_pct = round((adjusted - baseline) / baseline * 100, 2)
        else:
            row_pct = 0.0

        results.append(WhatIfPoint(
            date=fp.date,
            baseline_sales=round(baseline, 4),
            adjusted_sales=round(adjusted, 4),
            difference=difference,
            change_percent=row_pct,
        ))

    total_baseline = round(sum(r.baseline_sales for r in results), 4)
    total_adjusted = round(sum(r.adjusted_sales for r in results), 4)
    total_difference = round(total_adjusted - total_baseline, 4)

    return WhatIfResponse(
        product=request.product,
        change_percent=request.change_percent,
        total_baseline=total_baseline,
        total_adjusted=total_adjusted,
        total_difference=total_difference,
        disruption_days=disruption_days,
        results=results,
    )
