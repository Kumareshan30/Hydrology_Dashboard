// components/sections/Hydrometeorology.tsx
import { useEffect } from "react";
import dynamic from "next/dynamic";
import PlotCard from "@/components/PlotCard";
import { Slider } from "@/components/ui/slider";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
    activeTab: string;
    plotData: { data: any[]; layout: any } | null;
    sliderRange: [number, number];
    setSliderRange: (r: [number, number]) => void;
}

export default function HydrometeorologySection({
    activeTab,
    plotData,
    sliderRange,
    setSliderRange,
}: Props) {
    if (!plotData) return null;

    // Time Series & Climatic Indices
    if (["Time Series", "Climatic Indices"].includes(activeTab)) {
        let description: React.ReactNode = null;
        switch (activeTab) {
            case "Time Series":
                description = (
                    <div className="prose max-w-none mb-4">
                        <p>
                            <strong>Hydrometeorological Time Series:</strong><br />
                            Explore the full daily and monthly history of rainfall (or streamflow) at your selected station.
                        </p>
                        <ul>
                            <li>
                                <strong>Daily Precipitation (or Streamflow):</strong> Bars show every daily observation over the complete record (1910–2020).
                            </li>
                            <li>
                                <strong>Monthly Average:</strong> A smooth line/bar of monthly means highlighting seasonal cycles and anomalies.
                            </li>
                            <li>
                                <strong>Monthly Distribution:</strong> Boxplots for each calendar month, revealing variability, outliers, and extreme events.
                            </li>
                        </ul>
                        <p><em>How to use:</em></p>
                        <ul>
                            <li>
                                Spot long‑term shifts in seasonality or changes in variability (e.g. wetter winters vs. drier summers).
                            </li>
                            <li>
                                Identify extreme events (spikes or deep troughs) that may drive flood or drought analyses.
                            </li>
                            <li>
                                Compare monthly spread to understand which months are most volatile—critical for seasonal planning.
                            </li>
                        </ul>
                        <p>
                            <em>Why it matters for Australia:</em>
                            From tropical monsoons in the north to Mediterranean winters in the south, this view helps you
                            tailor water management and agricultural practices to each catchment’s unique hydro‑climate.
                        </p>
                    </div>
                );
                break;

            case "Climatic Indices":
                description = (
                    <div className="prose max-w-none mb-4">
                        <p>
                            <strong>Distribution of Aridity by State:</strong><br />
                            This boxplot shows the long‐term aridity index (PET / Precipitation) for each Australian state.
                            Higher values indicate drier climates (more evaporative demand per unit rainfall).
                        </p>
                        <ul>
                            <li>
                                <strong>Median line</strong>: the “typical” aridity for each state.
                            </li>
                            <li>
                                <strong>Box (25 – 75%)</strong>: the interquartile range—how much aridity varies between relatively wet and dry years.
                            </li>
                            <li>
                                <strong>Whiskers &amp; outliers</strong>: extremes (very high aridity = droughty years; very low aridity = unusually wet years).
                            </li>
                        </ul>
                        <p><em>How to use:</em></p>
                        <ul>
                            <li>
                                <strong>Compare medians</strong> (e.g. WA and NT are among the driest; TAS is the least arid).
                            </li>
                            <li>
                                <strong>Assess variability</strong>: wide boxes (e.g. NT) mean big swings between dry and wet years.
                            </li>
                            <li>
                                <strong>Spot extremes</strong>: outliers flag years of exceptional drought or unusually high rainfall.
                            </li>
                        </ul>
                        <p>
                            <em>Why it matters for Australia: </em>
                            Understanding each region’s aridity distribution is critical for water‐supply planning,
                            agricultural risk management, and ecosystem resilience under a changing climate.
                        </p>
                    </div>
                );
                break;
        }
        return (
            <PlotCard title={activeTab}>
                {description}
                <Plot
                    data={plotData.data}
                    layout={plotData.layout}
                    style={{ width: "100%", height: "700px" }}
                    config={{ responsive: true }}
                />
            </PlotCard>
        );
    }

    // Extreme Value Analysis
    if (activeTab === "Extreme Value Analysis") {
        const descriptionExtremeValue = (
            <div className="prose max-w-none mb-4">
                <p>
                    <strong>Extreme Value Analysis:</strong><br />
                    Identify and compare high‐magnitude precipitation events by focusing on exceedances above a chosen percentile threshold.
                </p>
                <ul>
                    <li>
                        <strong>POT Exceedances</strong>:
                        Red dots mark days where daily precipitation exceeds the 95<sup>th</sup> percentile (horizontal dashed line).
                    </li>
                    <li>
                        <strong>Seasonal Maxima</strong>:
                        For each year, bars show the maximum daily precipitation in each season—DJF (summer), MAM (autumn), JJA (winter) and SON (spring).
                    </li>
                </ul>
                <p><em>How to use:</em></p>
                <ul>
                    <li>
                        <strong>Select period</strong> with the slider to zoom in on any historical window (e.g. 1977–2020).
                    </li>
                    <li>
                        <strong>Assess frequency</strong>—note if extreme exceedances are becoming more or less common over time.
                    </li>
                    <li>
                        <strong>Compare seasons</strong>—see which season tends to produce the largest extremes at your catchment.
                    </li>
                </ul>
                <p>
                    <em>Why it matters for Australia: </em><br />
                    Draining catchments and flood‐risk infrastructure need design storms based on real extremes; this analysis helps planners quantify how often and when the worst rainfalls occur.
                </p>
            </div>
        );
        return (
            <PlotCard title="Extreme Value Analysis">
                {descriptionExtremeValue}
                <div className="px-6 pb-4 pt-15 flex items-center space-x-4">
                    <span className="font-medium">{sliderRange[0]}</span>
                    <Slider
                        className="flex-1"
                        value={sliderRange}
                        min={1911}
                        max={2020}
                        step={1}
                        onValueChange={(vals) => {
                            const [a, b] = vals as [number, number];
                            if (a <= b) setSliderRange([a, b]);
                        }}
                    />
                    <span className="font-medium">{sliderRange[1]}</span>
                </div>
                <Plot
                    data={plotData.data}
                    layout={{ ...plotData.layout, autosize: true }}
                    style={{ width: "100%", height: "700px" }}
                    config={{ responsive: true }}
                />
            </PlotCard>
        );
    }

    // Modeling & Forecasting
    if (activeTab === "Modeling & Forecasting") {
        const descriptionModelingForecasting = (
            <div className="prose max-w-none mb-4">
                <p>
                    <strong>Modeling &amp; Forecasting:</strong><br />
                    Fit an ARIMA(1,1,1)[12] model to your selected monthly precipitation record and compare in‑sample forecasts against the observed data.
                </p>
                <ul>
                    <li>
                        <strong>Observed vs. Forecast</strong>:
                        Blue = actual monthly precipitation;
                        Red = ARIMA in‑sample predictions.
                    </li>
                    <li>
                        <strong>Training Window</strong>:
                        Use the slider to choose your start and end years for model fitting (e.g. 2000–2020).
                    </li>
                    <li>
                        <strong>Model Specs</strong>:
                        ARIMA(p,d,q) = (1,1,1) with seasonal period = 12 to capture annual cycles.
                    </li>
                </ul>
                <p><em>How to use:</em></p>
                <ul>
                    <li>
                        <strong>Assess fit</strong>:
                        Are the seasonal peaks and troughs matched well?
                    </li>
                    <li>
                        <strong>Spot biases</strong>:
                        Look for systematic under‑ or over‑prediction during extreme dry or wet months.
                    </li>
                    <li>
                        <strong>Experiment</strong>:
                        Slide the window to see how expanding or shortening the training period affects forecast skill.
                    </li>
                </ul>
                <p>
                    <em>Why it matters for Australia: </em>
                    Reliable monthly forecasts help water resource planners anticipate storage needs, manage drought risk, and design resilient water supply systems in our highly variable climate.
                </p>
            </div>
        );
        return (
            <PlotCard title="Modeling & Forecasting">
                {descriptionModelingForecasting}
                <div className="px-6 pt-15 pb-4 flex items-center space-x-4">
                    <span className="font-medium">{sliderRange[0]}</span>
                    <Slider
                        className="flex-1"
                        value={sliderRange}
                        min={1911}
                        max={2020}
                        step={1}
                        onValueChange={(vals) => {
                            const [a, b] = vals as [number, number];
                            if (a <= b) setSliderRange([a, b]);
                        }}
                    />
                    <span className="font-medium">{sliderRange[1]}</span>
                </div>
                <Plot
                    data={plotData.data}
                    layout={{ ...plotData.layout, autosize: true }}
                    style={{ width: "100%", height: "700px" }}
                    config={{ responsive: true }}
                />
            </PlotCard>
        );
    }

    return null;
}
