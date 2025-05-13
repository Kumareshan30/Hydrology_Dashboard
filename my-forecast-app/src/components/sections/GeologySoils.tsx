// components/sections/GeologySoils.tsx
import dynamic from "next/dynamic";
import PlotCard from "@/components/PlotCard";
import type { Data, Layout } from "plotly.js";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
    activeTab: string;
    plotData: { data: Data[]; layout: Partial<Layout> } | null;
    stateList: string[];
    stateFilter: string;
    setStateFilter: (s: string) => void;
}

export default function GeologySoilsSection({
    activeTab,
    plotData,
    stateList,
    stateFilter,
    setStateFilter,
}: Props) {
    if (!plotData) return null;

    if (["Ksat Boxplot", "Clay vs Sand Scatter", "Proportions Stacked Bar", "PCA Biplot"].includes(activeTab)) {
        // build a short blurb per chart
        let description: React.ReactNode = null;
        switch (activeTab) {
            case "Ksat Boxplot":
                description = (
                    <div className="prose max-w-none mb-4">
                        <p>
                            <strong>Ksat Boxplot:</strong><br />
                            This boxplot summarizes the distribution of saturated hydraulic conductivity (Ksat)
                            for each primary geology class across the CAMELS‑AUS catchments.
                        </p>
                        <ul>
                            <li>
                                <strong>Box:</strong> 25th–75th percentile of Ksat (mm h⁻¹)
                            </li>
                            <li>
                                <strong>Line:</strong> median value
                            </li>
                            <li>
                                <strong>Whiskers:</strong> range (±1.5 × IQR)
                            </li>
                            <li>
                                <strong>Dots:</strong> outlier catchments with exceptionally low or high Ksat
                            </li>
                        </ul>
                        <p>
                            <em>How to use:</em>
                        </p>
                        <ul>
                            <li>
                                <strong>Compare medians</strong> to see which geologies drain fastest versus slowest.
                            </li>
                            <li>
                                <strong>Gauge variability</strong>—wide boxes indicate high within‑class spread.
                            </li>
                            <li>
                                <strong>Identify outliers</strong> for targeted field surveys or model calibration.
                            </li>
                        </ul>
                        <p>
                            <em>Why it matters for Australia: </em>
                            Soil infiltration rates vary dramatically—knowing typical and extreme Ksat values helps
                            predict groundwater recharge in arid zones and flood risks in clay‑rich catchments.
                        </p>
                    </div>
                );
                break;

            case "Clay vs Sand Scatter":
                description = (
                    <div className="prose max-w-none mb-4">
                        <p>
                            <strong>Clay vs Sand Scatter:</strong><br />
                            A scatterplot of catchment‑average clay % (x‑axis) versus sand % (y‑axis), colored by primary geology.
                        </p>
                        <ul>
                            <li>
                                <strong>Downward trend:</strong> more clay means less sand, and vice versa.
                            </li>
                            <li>
                                <strong>Color coding:</strong> primary geology classes occupy distinct texture bands.
                            </li>
                        </ul>
                        <p>
                            <em>How to use:</em>
                        </p>
                        <ul>
                            <li>
                                <strong>Group catchments</strong> with similar clay:sand ratios for comparative modeling.
                            </li>
                            <li>
                                <strong>Spot outliers</strong>—texture extremes that may require custom parameter sets.
                            </li>
                            <li>
                                <strong>Stratify analyses</strong> into “coarse,” “mixed,” and “fine” texture classes.
                            </li>
                        </ul>
                        <p>
                            <em>Why it matters for Australia: </em>
                            From sandy arid catchments to clayey temperate basins, soil texture drives infiltration,
                            runoff generation, and drought resilience—critical for water resource planning.
                        </p>
                    </div>
                );
                break;

            case "Proportions Stacked Bar":
                description = (
                    <div className="prose max-w-none mb-4">
                        <p>
                            <strong>Proportions Stacked Bar:</strong><br />
                            Each bar shows the mean fraction of secondary geology types within a primary geology class.
                        </p>
                        <ul>
                            <li>
                                <strong>Total = 100 %:</strong> each slice represents one secondary rock type’s average proportion.
                            </li>
                            <li>
                                <strong>Color slices:</strong> distinguish different secondary lithologies.
                            </li>
                        </ul>
                        <p>
                            <em>How to use:</em>
                        </p>
                        <ul>
                            <li>
                                <strong>Assess complexity:</strong> uniform bars = simple geology, mixed bars = heterogeneous catchment.
                            </li>
                            <li>
                                <strong>Select clusters:</strong> choose catchments with similar secondary‑rock mixtures.
                            </li>
                            <li>
                                <strong>Regionalize parameters:</strong> tailor hydrological models to dominant lithologies.
                            </li>
                        </ul>
                        <p>
                            <em>Why it matters for Australia: </em>
                            Many catchments span lithological transitions (e.g. sedimentary–metamorphic); knowing the average
                            mix informs sediment transport, water chemistry, and resource management in complex basins.
                        </p>
                    </div>
                );
                break;

            case "PCA Biplot":
                description = (
                    <div className="prose max-w-none mb-4">
                        <p>
                            <strong>PCA Biplot:</strong><br />
                            This biplot projects each catchment into the first two principal component axes, which together explain the majority of variability in soil‐proportion data:
                        </p>
                        <ul>
                            <li>
                                <strong>PC1</strong> often captures the dominant gradient (e.g. a sand–clay continuum). Catchments at the right end
                                are sandier; those at the left end are clayey or silty.
                            </li>
                            <li>
                                <strong>PC2</strong> may reflect a secondary texture or geology axis (e.g. ratio of carbonates vs. metamorphic rock fragments).
                            </li>
                        </ul>
                        <p>
                            <em>How to use:</em>
                        </p>
                        <ul>
                            <li>
                                <strong>Identify clusters</strong> of catchments that plot close together—these have very similar soil make‐up and could be modeled with the same parameter set.
                            </li>
                            <li>
                                <strong>Spot outliers</strong> that fall far from the main clusters—these may be geologically unique or require a custom modeling approach.
                            </li>
                            <li>
                                <strong>Interpret loadings</strong> (if shown): vector arrows indicate which soil types drive the axes, helping you link physical soil properties back to catchment behavior (e.g. high sand loadings → fast drainage).
                            </li>
                        </ul>
                        <p>
                            <em>Why it matters for Australia:</em> so many of our catchments span arid to temperate climates—this tool helps you group arid sandy‐soil catchments separately from more clay‐rich temperate catchments, improving regional hydrological predictions and parameter regionalization.
                        </p>
                    </div>
                );
                break;
        }

        return (
            <PlotCard title={activeTab}>
                {description}
                <Plot
                    data={plotData!.data}
                    layout={plotData!.layout}
                    style={{ width: "100%", height: "500px" }}
                    config={{ responsive: true }}
                />
            </PlotCard>
        );
    }

    // K‑Means
    if (activeTab === "K‑Means Clusters") {
        const descriptionKMeans = (
            <div className="prose max-w-none mb-4">
                <p>
                    <strong>K‑Means Clusters:</strong><br />
                    This scatter in PCA space shows catchments grouped into k clusters based on their soil and hydrological attributes.
                </p>
                <ul>
                    <li>
                        <strong>Points:</strong> individual catchments plotted on principal component axes (PC1, PC2).
                    </li>
                    <li>
                        <strong>Colors:</strong> cluster membership (e.g. “Sand‑Rich”, “High‑Ksat”, “Clay‑Rich”, “Sed‑Volcanic”).
                    </li>
                </ul>
                <p>
                    <em>How to use:</em>
                </p>
                <ul>
                    <li>
                        <strong>Compare clusters:</strong> identify groups of catchments sharing similar combined attributes.
                    </li>
                    <li>
                        <strong>Customize models:</strong> assign cluster‑specific parameters rather than per‑catchment.
                    </li>
                    <li>
                        <strong>Spot reassignment:</strong> catchments near boundary may be sensitive to clustering choice (k).
                    </li>
                </ul>
                <p>
                    <em>Why it matters for Australia: </em>
                    Rapidly categorize hundreds of diverse Australian catchments into a few representative types,
                    accelerating regionalization of hydrological models across arid, semi‑arid, and temperate zones.
                </p>
            </div>
        );
        return (
            <PlotCard title="K‑Means Clusters">
                {descriptionKMeans}
                <Plot
                    data={plotData.data}
                    layout={plotData.layout}
                    style={{ width: "100%", height: "500px" }}
                    config={{ responsive: true }}
                />
            </PlotCard>
        );
    }

    // Hierarchical Heatmap
    if (activeTab === "Hierarchical Heatmap") {
        const descriptionHeatmap = (
            <div className="prose max-w-none mb-4">
                <p>
                    <strong>Hierarchical Heatmap:</strong><br />
                    A heatmap of scaled soil attributes, hierarchically clustered by similarity, for catchments in a selected state.
                </p>
                <ul>
                    <li>
                        <strong>Rows:</strong> individual gauging stations (with ID, name, state).
                    </li>
                    <li>
                        <strong>Columns:</strong> soil and landscape attributes (e.g. geology proportions, ksat, texture indices).
                    </li>
                    <li>
                        <strong>Colors:</strong> z‑score of each attribute—bright = high, dark = low.
                    </li>
                </ul>
                <p>
                    <em>How to use:</em>
                </p>
                <ul>
                    <li>
                        <strong>Cluster patterns:</strong> find groups of catchments with similar multivariate signatures.
                    </li>
                    <li>
                        <strong>Attribute drivers:</strong> see which attributes (columns) most differentiate clusters.
                    </li>
                    <li>
                        <strong>State filter:</strong> focus on regional subsets to capture local geological or climatic regimes.
                    </li>
                </ul>
                <p>
                    <em>Why it matters for Australia: </em>
                    Our continent spans extreme diversity—this tool reveals coherent sub‑regions (e.g. Tasmanian volcanic soils vs. arid central Australia),
                    guiding targeted management and model transferability within states.
                </p>
            </div>
        );
        return (
            <>
                <div className="px-6 pb-4 flex items-center space-x-4">
                    <label htmlFor="state-filter" className="font-medium text-gray-700">
                        Filter by State:
                    </label>
                    <select
                        id="state-filter"
                        className="border rounded px-2 py-1"
                        value={stateFilter}
                        onChange={(e) => setStateFilter(e.target.value)}
                    >
                        <option value="">All States</option>
                        {stateList.map((st) => (
                            <option key={st} value={st}>
                                {st}
                            </option>
                        ))}
                    </select>
                </div>
                <PlotCard title="Hierarchical Heatmap">
                    {descriptionHeatmap}
                    <Plot
                        data={plotData.data}
                        layout={plotData.layout}
                        style={{ width: "100%", height: `${plotData.layout.height}px` }}
                        config={{ responsive: true }}
                    />
                </PlotCard>
            </>
        );
    }

    return null;
}
