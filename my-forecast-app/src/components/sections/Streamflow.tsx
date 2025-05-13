// components/sections/Streamflow.tsx
import dynamic from "next/dynamic";
import type { ComponentType } from "react";
import PlotCard from "@/components/PlotCard";
import MannKendallResult from "@/components/MannKendallResult";
import type { Data, Layout } from "plotly.js";
import type { PlotParams } from "react-plotly.js";

const Plot: ComponentType<PlotParams> = dynamic(() => import("react-plotly.js"), { ssr: false });

interface MannKendallData {
  trend: string
  p_value: number
  test_statistic: number
  z: number
  interpretation?: string
}

interface Props {
    activeTab: string;
    selectedStation: string;
    plotData: { data: Data[]; layout: Partial<Layout> } | null;    
    mannKendallData: MannKendallData | null;
    sliderRange: [number, number];
    setSliderRange: (r: [number, number]) => void;
}

export default function StreamflowSection({
    activeTab,
    plotData,
    mannKendallData,
}: Props) {
    // Monthly Flow, Trend Line, ARIMA Decomp all just show a 700px chart
    if (["Monthly Flow", "Trend Line"].includes(activeTab)) {

        let blurb = ""
        switch (activeTab) {
            case "Monthly Flow":
                blurb = `This chart shows the long‑term monthly average streamflow at the Station. \
Use it to see seasonal cycles (wet vs dry months) and year‑to‑year variability.`
                break
            case "Trend Line":
                blurb = `Here we plot each monthly observation (blue dots) together with a fitted trend line (red). \
A slope above zero means streamflow has generally been rising; below zero means a long‑term decline.`
                break
        }

        return plotData ? (
            <>
                <div className="max-w-4xl mx-auto px-6 py-4">
                    <p className="text-gray-700">{blurb}</p>
                </div>
                <PlotCard title={activeTab}>
                    <Plot
                        data={plotData.data}
                        layout={plotData.layout}
                        style={{ width: "100%", height: "700px" }}
                        config={{ responsive: true }}
                    />
                </PlotCard>
            </>) : null;
    }

    if (activeTab === "ARIMA Decomposition" &&
        plotData != null) {
        return (
            <PlotCard title="ARIMA Decomposition">
                {/* Blurb */}
                <div className="prose max-w-4xl mx-auto px-6 mb-6 p">
                    <p>
                        The ARIMA decomposition splits your historical streamflow into four
                        interpretable components:
                    </p>
                    <ul>
                        <li>
                            <strong>Observed</strong> – the raw monthly values you actually
                            measured.
                        </li>
                        <li>
                            <strong>Trend</strong> – a smooth, long‑term baseline that
                            highlights gradual increases or declines in flow over decades.
                        </li>
                        <li>
                            <strong>Seasonal</strong> – the consistent within‑year cycles
                            (e.g. winter highs, summer lows) driven by Australia’s wet/dry
                            seasons.
                        </li>
                        <li>
                            <strong>Residual</strong> – the irregular “leftover” fluctuations
                            and anomalies not explained by trend or seasonality (e.g. extreme
                            floods or droughts).
                        </li>
                    </ul>
                    <p>
                        <strong>Why it matters</strong>
                    </p>
                    <ul>
                        <li>
                            Break out the slow drift (trend) from repeating seasonal behavior.
                        </li>
                        <li>
                            Plan for regular seasonal peaks and troughs when designing
                            reservoirs or allocating water rights.
                        </li>
                        <li>
                            Spot out‑of‑pattern events (residuals) that may signal data issues,
                            unusual climate episodes, or the need for a more complex model.
                        </li>
                    </ul>
                </div>
                {/* Plot */}
                <Plot
                    data={plotData.data}
                    layout={plotData.layout}
                    style={{ width: "100%", height: "700px" }}
                    config={{ responsive: true }}
                />
            </PlotCard>
        );
    }

    if (
        activeTab === "ARIMA Forecast" &&
        plotData != null &&
        mannKendallData != null
    ) {
        return (
            <>
                <div className="max-w-4xl mx-auto px-1 py-1">
                    <p className="text-gray-700 mb-2">
                        Here you see the last 10 years of observed monthly streamflow at the station
                        overlaid with a 12‑month ARIMA
                        forecast (red) and its 95 % confidence bounds (shaded). Use this to
                        anticipate seasonal water availability, plan reservoir releases,
                        or flag emerging drought/flood risk.
                    </p>

                </div>

                <PlotCard title="Streamflow Forecast & Trend Test using ARIMA">
                    {/* ARIMA forecast chart */}
                    <Plot
                        data={plotData.data}
                        layout={plotData.layout}
                        style={{ width: "100%", height: "700px" }}
                        config={{ responsive: true }}
                    />
                    <div className="max-w-4xl mx-auto px-1 py-1">
                        <p className="text-gray-700">
                            Below, the Mann–Kendall test tells you if there’s a statistically
                            significant upward or downward trend in the forecasted series —
                            useful for assessing long‑term changes due to climate or land‑use
                            impacts.
                        </p>
                    </div>
                    {/* Mann–Kendall results below */}
                    <div className="mt-6">
                        <MannKendallResult data={mannKendallData} />
                    </div>
                </PlotCard>
            </>
        );
    }

    return null;
}
