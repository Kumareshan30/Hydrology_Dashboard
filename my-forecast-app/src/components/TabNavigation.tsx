import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

export type Tab =
  | "Monthly Flow"
  | "Trend Line"
  | "ARIMA Decomposition"
  | "ARIMA Forecast"
  | "Feature Importance RF"
  | "Feature Importance XGB"
  | "Anomaly Detection"
  | "Geospatial Analysis"
  | "Density Map"
  | "Time Series"
  | "Climatic Indices"
  | "Modeling & Forecasting"
  | "Extreme Value Analysis"
  | "Ksat Boxplot"
  | "Clay vs Sand Scatter"
  | "Proportions Stacked Bar"
  | "PCA Biplot"
  | "K‑Means Clusters"
  | "Hierarchical Heatmap";

const tabs: Tab[] = [
  "Monthly Flow",
  "Trend Line",
  "ARIMA Decomposition",
  "ARIMA Forecast",
  "Feature Importance RF",
  "Feature Importance XGB",
  "Anomaly Detection",
  "Geospatial Analysis",
  "Density Map",
  "Time Series",
  "Climatic Indices",
  "Modeling & Forecasting",
  "Extreme Value Analysis",
  "Ksat Boxplot",
  "Clay vs Sand Scatter",
  "Proportions Stacked Bar",
  "PCA Biplot",
  "K‑Means Clusters",
  "Hierarchical Heatmap",
];

export default function TabNavigation({
  activeTab,
  setActiveTab,
}: {
  activeTab: Tab;
  setActiveTab: (tab: Tab) => void;
}) {
  return (
    <Tabs value={activeTab} className="w-full" onValueChange={(val: Tab) => setActiveTab(val)}>
      <TabsList className="flex flex-wrap justify-center gap-2">
        {tabs.map((tab) => (
          <TabsTrigger key={tab} value={tab}>
            {tab}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
