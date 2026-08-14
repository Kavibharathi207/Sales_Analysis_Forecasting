from fastapi import APIRouter
from backend.schemas.whatif import WhatIfRequest, WhatIfResponse
from backend.services.whatif_service import run_what_if

router = APIRouter(prefix="/api/what-if", tags=["What-If Analysis"])


@router.post("", response_model=WhatIfResponse)
def what_if(request: WhatIfRequest):
    """
    Apply a scenario to a forecast series.

    - change_percent: uniform % shift (e.g. 20 = +20%, -15 = -15%)
    - disruption_start/end: date range where supply is zeroed out
    """
    return run_what_if(request)
