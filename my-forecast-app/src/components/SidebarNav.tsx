// components/SidebarNav.tsx
import Link from "next/link";
import { useRouter } from "next/router";
import { Button } from "@/components/ui/button";

interface SidebarNavProps {
  currentTab: string;
  onSelectTab: (tab: string) => void;
}

const tabGroups: Record<string, string[]> = {
  "Streamflow Analysis": [
    "Monthly Flow",
    "Trend Line",
    "ARIMA Decomposition",
    "ARIMA Forecast",
  ],
  "Geology & Soil Analysis": [
    "Ksat Boxplot",
    "Clay vs Sand Scatter",
    "Proportions Stacked Bar",
    "PCA Biplot",
    "K‑Means Clusters",
    "Hierarchical Heatmap",
  ],
  "Catchment Analysis": ["Geospatial Analysis", "Density Map"],
  "Hydrometeorology": [
    "Time Series",
    "Climatic Indices",
    "Modeling & Forecasting",
    "Extreme Value Analysis",
  ],
  // "Statistics": ["Feature Importance RF", "Feature Importance XGB", "Anomaly Detection"],
};

export default function SidebarNav({ currentTab, onSelectTab }: SidebarNavProps) {
  const { pathname } = useRouter();
  const onDashboard = pathname === "/dashboard";

  return (
    <div className="w-56 min-h-screen bg-black text-white border-r border-gray-800 p-4">
      {/* Home / Dashboard */}
      <div className="mb-6 space-y-1">
        <Link href="/">
          <Button
            variant={pathname === "/" ? "secondary" : "ghost"}
            className={`w-full justify-start ${pathname === "/" ? "bg-white text-black font-medium" : "hover:bg-gray-800"}`}
          >
            Home
          </Button>
        </Link>
        <Link href="/dashboard">
          <Button
            variant={onDashboard ? "secondary" : "ghost"}
            className={`w-full justify-start ${onDashboard ? "bg-white text-black font-medium" : "hover:bg-gray-800"}`}
          >
            Dashboard
          </Button>
        </Link>
      </div>

      {/* Only show the real nav when we’re on /dashboard */}
      {onDashboard &&
        Object.entries(tabGroups).map(([section, tabs]) => (
          <div key={section} className="mb-6">
            <p className="text-sm font-semibold text-gray-400 uppercase mb-2">{section}</p>
            <div className="space-y-1">
              {tabs.map((tab) => (
                <Button
                  key={tab}
                  variant={currentTab === tab ? "secondary" : "ghost"}
                  className={`w-full justify-start ${
                    currentTab === tab
                      ? "bg-white text-black font-medium"
                      : "hover:bg-gray-800"
                  }`}
                  onClick={() => onSelectTab(tab)}
                >
                  {tab}
                </Button>
              ))}
            </div>
          </div>
        ))}
    </div>
);
}
