interface MannKendallData {
  trend: string
  p_value: number
  test_statistic: number
  z: number
  interpretation?: string
}

export default function MannKendallResult({ data }: { data: MannKendallData }) {
  return (
    <div className="max-w-xl mx-auto bg-white shadow rounded p-6 border border-gray-200">
      <h2 className="text-xl font-bold text-black text-center mb-4">
        Mann‑Kendall Test Forecast
      </h2>

      {data.interpretation && (
        <p className="italic text-gray-600 mb-4 text-center">
          {data.interpretation}
        </p>
      )}

      <ul className="text-base space-y-1 text-gray-700">
        <li>
          <strong>Trend:</strong> {data.trend}
        </li>
        <li>
          <strong>p‑value:</strong> {data.p_value.toFixed(3)}
        </li>
        <li>
          <strong>Test Statistic (S):</strong> {data.test_statistic}
        </li>
        <li>
          <strong>z‑score:</strong> {data.z.toFixed(2)}
        </li>
      </ul>
    </div>
  )
}

export interface MannKendallResultData {
  trend: string;
  p_value: number;
  test_statistic: number;
  z: number;
}