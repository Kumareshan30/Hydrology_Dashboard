from fastapi import APIRouter
from ..models.SummaryMetrics import SummaryMetrics  # note the double-dot for going up one directory

router = APIRouter()

@router.get("/overview", response_model=SummaryMetrics)
async def get_overview_metrics():
    # ...
    return SummaryMetrics(
        totalCatchments=222,
        averageCatchmentArea=1200.0,
        averageStreamflow=50.0,
        totalPrecipitation=200.0,
    )
