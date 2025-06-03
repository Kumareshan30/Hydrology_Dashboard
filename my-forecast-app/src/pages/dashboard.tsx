// pages/dashboard.tsx
import React, { useEffect, useState } from "react";
import type { Tab } from "@/components/TabNavigation";
import StationSelect from "@/components/StationSelect";
import SidebarNav from "@/components/SidebarNav";
import { motion } from "framer-motion";
import { Skeleton } from "@/components/ui/skeleton";
import { MannKendallResultData } from "@/components/MannKendallResult";
import StreamflowSection from "@/components/sections/Streamflow";
import CatchmentSection from "@/components/sections/Catchment";
import HydrometeorologySection from "@/components/sections/Hydrometeorology";
import GeologySoilsSection from "@/components/sections/GeologySoils";
import type { Data, Layout, Config  } from "plotly.js";

interface PlotData {
  data: Data[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
}


// Debounce hook
function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState<T>(value);
  useEffect(() => {
    const h = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(h);
  }, [value, delay]);
  return debounced;
}

const endpointMapping: Record<Tab, (s: string) => string> = {
  "Monthly Flow": (s) => `https://hydrologydashboard-production.up.railway.app/monthly_flow_plot?station_id=${s}`,
  "Trend Line": (s) => `https://hydrologydashboard-production.up.railway.app/trend_line_plot?station_id=${s}`,
  "ARIMA Decomposition": (s) =>
    `https://hydrologydashboard-production.up.railway.app/arima_decomposition_plot?station_id=${s}`,
  "ARIMA Forecast": (s) =>
    `https://hydrologydashboard-production.up.railway.app/arima_forecast_plot?station_id=${s}&steps=12`,
  "Feature Importance RF": () => `https://hydrologydashboard-production.up.railway.app/feature_importance_rf`,
  "Feature Importance XGB": () => `https://hydrologydashboard-production.up.railway.app/feature_importance_xgb`,
  "Anomaly Detection": (s) =>
    `https://hydrologydashboard-production.up.railway.app/anomaly_detection?station_id=${s}`,
  "Geospatial Analysis": () => `https://hydrologydashboard-production.up.railway.app/geospatial_plot`,
  "Density Map": () => `https://hydrologydashboard-production.up.railway.app/density_map`,
  "Time Series": (s) =>
    `https://hydrologydashboard-production.up.railway.app/hydrometeorology_timeseries?station_id=${s}`,
  "Climatic Indices": () => `https://hydrologydashboard-production.up.railway.app/hydrometeorology_indices`,
  "Modeling & Forecasting": (s) =>
    `https://hydrologydashboard-production.up.railway.app/hydrometeorology_modeling?station_id=${s}`,
  "Extreme Value Analysis": (s) =>
    `https://hydrologydashboard-production.up.railway.app/hydrometeorology_extreme?station_id=${s}`,
  "Ksat Boxplot": () => `https://hydrologydashboard-production.up.railway.app/soil/ksat_boxplot`,
  "Clay vs Sand Scatter": () => `https://hydrologydashboard-production.up.railway.app/soil/clay_sand_scatter`,
  "Proportions Stacked Bar": () => `https://hydrologydashboard-production.up.railway.app/soil/prop_stacked_bar`,
  "PCA Biplot": () => `https://hydrologydashboard-production.up.railway.app/soil/pca_biplot`,
  "K‑Means Clusters": () => `https://hydrologydashboard-production.up.railway.app/soil/kmeans`,
  "Hierarchical Heatmap": () => `https://hydrologydashboard-production.up.railway.app/soil/hierarchical`,
};

const hideStationDropdownTabs: Tab[] = [
  "Feature Importance RF",
  "Feature Importance XGB",
  "Geospatial Analysis",
  "Density Map",
  "Climatic Indices",
  "Ksat Boxplot",
  "Clay vs Sand Scatter",
  "Proportions Stacked Bar",
  "PCA Biplot",
  "K‑Means Clusters",
  "Hierarchical Heatmap",
];

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<Tab>("Monthly Flow");
  const [plotData, setPlotData] = useState<PlotData | null>(null);
  const [mannKendallData, setMannKendallData] =
    useState<MannKendallResultData | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedStation, setSelectedStation] = useState("912101A");

  const [sliderRange, setSliderRange] = useState<[number, number]>([
    1911,
    2020,
  ]);
  const debouncedRange = useDebounce(sliderRange, 500);

  const [stateList, setStateList] = useState<string[]>([]);
  const [stateFilter, setStateFilter] = useState<string>("");

  const showStationDropdown = !hideStationDropdownTabs.includes(activeTab);

  // Fetch available states for the hierarchical heatmap
  useEffect(() => {
    if (activeTab === "Hierarchical Heatmap") {
      fetch("https://hydrologydashboard-production.up.railway.app/soil/hierarchical/states")
        .then((r) => r.json())
        .then(setStateList)
        .catch(console.error);
    }
  }, [activeTab]);

  // Main data‐fetch effect
  useEffect(() => {
    setLoading(true);
    setPlotData(null);
    setMannKendallData(null);

    let url: string;

    if (activeTab === "Modeling & Forecasting") {
      const [start, end] = debouncedRange;
      url = endpointMapping[activeTab](selectedStation) + `&start_year=${start}&end_year=${end}`;
    } else if (activeTab === "Extreme Value Analysis") {
      const [start, end] = debouncedRange;
      url = endpointMapping[activeTab](selectedStation) + `&start_year=${start}&end_year=${end}`;
    } else if (activeTab === "Hierarchical Heatmap") {
      url = endpointMapping[activeTab](selectedStation);
      if (stateFilter) {
        url += `?state=${encodeURIComponent(stateFilter)}`;
      }
    } else {
      url = endpointMapping[activeTab](selectedStation);
    }

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error("Network response was not OK");
        return res.json();
      })
      .then((data) => {
        if (activeTab === "ARIMA Forecast") {
          setPlotData(data.plot as PlotData);
          setMannKendallData(data.mannkendall as MannKendallResultData);
        } else {
          setPlotData(data as PlotData);
        }
      })
      .catch((err) => console.error("Fetch error:", err))
      .finally(() => setLoading(false));
  }, [activeTab, selectedStation, debouncedRange, stateFilter]);

  return (
    <div className="flex min-h-screen bg-gray-50">
      <SidebarNav
        currentTab={activeTab}
        onSelectTab={(tab) => setActiveTab(tab as Tab)}
      />

      <div className="flex-1">
        {/* Header */}
        <div className="w-full bg-black py-6 px-8 text-white text-center">
          <h1 className="text-5xl font-bold">Hydrology Dashboard</h1>
        </div>

        {/* Station selector */}
        <div className="p-6">
          {showStationDropdown && (
            <div className="text-center space-y-2">
              <p className="text-gray-700 font-medium">Select Station</p>
              <StationSelect
                value={selectedStation}
                onChange={setSelectedStation}
              />
            </div>
          )}
        </div>

        {/* Loading skeleton */}
        {loading && (
          <div className="space-y-4 max-w-5xl mx-auto px-6 py-8">
            <Skeleton className="h-6 w-64 mx-auto" />
            <Skeleton className="h-[500px] w-full rounded-md" />
          </div>
        )}

        {/* Main content */}
        {!loading && (
          <motion.div
            key={activeTab + stateFilter}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {/* Streamflow Analysis */}
            {[
              "Monthly Flow",
              "Trend Line",
              "ARIMA Decomposition",
              "ARIMA Forecast",
            ].includes(activeTab) && (
                <StreamflowSection
                  activeTab={activeTab}
                  selectedStation={selectedStation}
                  plotData={plotData}
                  mannKendallData={mannKendallData}
                  sliderRange={sliderRange}
                  setSliderRange={setSliderRange}
                />
              )}

            {/* Catchment Analysis */}
            {["Geospatial Analysis", "Density Map"].includes(activeTab) && (
              <CatchmentSection
                activeTab={activeTab}
                plotData={plotData}
              />
            )}

            {/* Hydrometeorology */}
            {[
              "Time Series",
              "Climatic Indices",
              "Extreme Value Analysis",
              "Modeling & Forecasting",
            ].includes(activeTab) && (
                <HydrometeorologySection
                  activeTab={activeTab}
                  plotData={plotData}
                  sliderRange={sliderRange}
                  setSliderRange={setSliderRange}
                />
              )}

            {/* Geology & Soil Analysis */}
            {[
              "Ksat Boxplot",
              "Clay vs Sand Scatter",
              "Proportions Stacked Bar",
              "PCA Biplot",
              "K‑Means Clusters",
              "Hierarchical Heatmap",
            ].includes(activeTab) && (
                <GeologySoilsSection
                  activeTab={activeTab}
                  plotData={plotData}
                  stateList={stateList}
                  stateFilter={stateFilter}
                  setStateFilter={setStateFilter}
                />
              )}
          </motion.div>
        )}
      </div>
    </div>
  );
}
