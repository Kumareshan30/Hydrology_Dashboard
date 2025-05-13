// components/sections/Catchment.tsx
import dynamic from "next/dynamic";
import PlotCard from "@/components/PlotCard";
import type { Data, Layout } from "plotly.js";
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  activeTab: string;
  plotData: { data: Data[]; layout: Partial<Layout> } | null;
}

export default function CatchmentSection({ activeTab, plotData }: Props) {
  if (!plotData) return null;

  if (activeTab === "Geospatial Analysis") {
    const descriptionGeospatial = (
      <div className="prose max-w-none mb-4">
        <p>
          <strong>Geospatial Analysis:</strong><br />
          An overview of where all catchments lie in Australia and how they’re distributed by state and drainage division.
        </p>
        <p>
          <em>Components:</em>
        </p>
        <ul>
          <li>
            <strong>Outlet Map:</strong>
            A Dot Map of catchment outlet locations (sized by catchment area) across the continent.
          </li>
          <li>
            <strong>Count by State:</strong>
            Bar chart showing the number of gauged catchments in each Australian state and territory.
          </li>
          <li>
            <strong>Area by Drainage Division:</strong>
            Boxplots of catchment areas grouped by major BOM drainage divisions.
          </li>
        </ul>
        <p>
          <em>How to use:</em>
        </p>
        <ul>
          <li>
            <strong>Spot sampling gaps:</strong> identify under‑represented regions where new monitoring sites could be added.
          </li>
          <li>
            <strong>Compare states:</strong> see which states have more gauges and ensure balanced coverage for regional analyses.
          </li>
          <li>
            <strong>Assess size variability:</strong> use the division‐level boxplots to understand the range of catchment sizes when setting model scales.
          </li>
        </ul>
        <p>
          <em>Why it matters for Australia: </em>
          With its vast geography and varied climates, mapping and summarizing catchments by location, jurisdiction,
          and scale is critical for national water resources planning, policy making, and modelling efforts.
        </p>
      </div>
    );
    return (
      <PlotCard title="Geospatial Analysis">
        {descriptionGeospatial}
        <Plot
          data={plotData.data}
          layout={plotData.layout}
          style={{ width: "100%", height: "2000px" }}
          config={{ responsive: true }}
        />
      </PlotCard>
    );
  }

  if (activeTab === "Density Map") {
    const descriptionDensityMap = (
      <div className="prose max-w-none mb-4">
        <p>
          <strong>Density Map of Catchments:</strong><br />
          A spatial heatmap showing where gauged catchments cluster across Australia, with color intensity
          reflecting the combined catchment area in each grid cell.
        </p>
        <p>
          <em>How to use:</em>
        </p>
        <ul>
          <li>
            <strong>Identify hotspots</strong> of high catchment density or large total area—these regions
            dominate national flow statistics and may drive large‐scale water management decisions.
          </li>
          <li>
            <strong>Spot data gaps</strong> in sparsely covered areas—key for planning future gauge installations
            or remote sensing campaigns.
          </li>
          <li>
            <strong>Weight sampling effort</strong> by area rather than count alone—helpful when your model
            sensitivity depends on catchment size.
          </li>
        </ul>
        <p>
          <em>Why it matters for Australia: </em>
          With vast, remote catchments in the outback and dense networks in coastal regions, this map
          highlights where our observational capacity is strongest—and where it needs bolstering—to support
          resilient, nation‑wide water resource management.
        </p>
      </div>
    );
    return (
      <PlotCard title="Density Map">
        {descriptionDensityMap}
        <Plot
          data={plotData.data}
          layout={plotData.layout}
          style={{ width: "100%", height: "500px" }}
          config={{ responsive: true }}
        />
      </PlotCard>
    );
  }

  return null;
}
