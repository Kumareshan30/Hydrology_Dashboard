# backend/app/models.py
from pydantic import BaseModel

class SummaryMetrics(BaseModel):
    totalCatchments: int
    averageCatchmentArea: float
    averageStreamflow: float
    totalPrecipitation: float
