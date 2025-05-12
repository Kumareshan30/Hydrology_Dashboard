// pages/index.tsx
import { useEffect, useState } from "react"
import { useRouter } from "next/router"
import { Button } from "@/components/ui/button"
import { Card, StatCard } from "@/components/ui/card"
import { Carousel } from "@/components/ui/carousel"

interface StationInfo {
  station_id: string
  station_name: string
  state_outlet: string
}

export default function Home() {
  const router = useRouter()

  // quick‐stats state
  const [numStations, setNumStations] = useState<number | null>(null)

  // constants for years
  const MIN_YEAR = 1910
  const MAX_YEAR = 2020

//   const keyFeatures = [
//     "Monthly & trend plots, ARIMA forecasting, Mann‑Kendall tests",
//     "KSAT boxplots, clay vs sand scatter, PCA & cluster heatmaps",
//     "Geospatial density maps & catchment attributes overview",
//     "Climatic signatures (aridity, seasonality, snow cover)",
//     "Adjustable‑range modeling & seasonal planning",
//     "Extreme value analysis for drought & flood risk",
//   ]
  
//   const useCases = [
//     "Water resource management (allocation, drought tracking)",
//     "Environmental planning (catchment health, land‑use design)",
//     "Climate change research (regional trend quantification)",
//     "Agricultural scheduling (align crops to soil moisture/rainfall)",
//   ]

  useEffect(() => {
    fetch("http://localhost:8000/stations")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load stations")
        return res.json()
      })
      .then((data: StationInfo[]) => {
        setNumStations(data.length)
      })
      .catch((err) => {
        console.error(err)
        setNumStations(0)
      })
  }, [])

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-50">
      {/* Header */}
      <header className="w-full bg-white py-12 text-center shadow-sm pt-20">
        <h1 className="text-5xl font-bold mb-2">
          Welcome to the Hydrology Dashboard
        </h1>
        <p className="text-gray-600 mb-6 mt-10 ml-25 mr-25">
        Discover and explore the CAMELS‑AUS v2 dataset: hydrometeorological time series and landscape attributes for hundreds of Australian catchments (1911–2020). Perfect for researchers, water managers, and policy makers looking to understand Australia’s rivers, soils, and climate. 
        </p>
        <button
          onClick={() => router.push("/dashboard")}
          className="px-6 py-3 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
        >
            Go to Dashboard
        </button>
      </header>

      {/* Quick‐Stats */}
      <section className="w-full max-w-5xl mt-12 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 px-4">
        <StatCard label="Catchments" value={numStations ?? "…"} />
        <StatCard label="Years Covered" value={`${MIN_YEAR} – ${MAX_YEAR}`} />
        <StatCard label="States/Territories Covered" value="6 (WA, NT, SA, QLD, NSW, VIC)" />
        <StatCard label="Total Catchment Area" value="580 000 km²" />
      </section>


      {/* Footer / Data Source & License */}
      <footer className="w-full mt-auto bg-white py-6 text-center text-sm text-gray-500 border-t">
        <p>
          Powered by the {" "}
          <a
            href="https://zenodo.org/records/13350616"
            target="_blank"
            rel="noreferrer"
            className="underline"
          >
            CAMELS‑AUS v2 dataset
          </a>{" "}
          (© University of Melbourne 2024, CC‑BY 4.0)
        </p>
        <p> CAMELS‑AUS v2: updated hydrometeorological timeseries and landscape attributes for an enlarged set of catchments in Australia.</p>
      </footer>
    </div>
  )
}

