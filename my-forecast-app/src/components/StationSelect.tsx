// components/StationSelect.tsx
import { useEffect, useState } from "react"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select"

interface Station {
  station_id: string
  station_name: string
  state_outlet: string
}

export default function StationSelect({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  const [stations, setStations] = useState<Station[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    fetch("http://localhost:8000/stations")
      .then((res) => {
        if (!res.ok) throw new Error(res.statusText)
        return res.json()
      })
      .then((data: Station[]) => {
        setStations(data)
      })
      .catch((err) => {
        console.error(err)
        setError("Failed to load stations")
      })
      .finally(() => setLoading(false))
  }, [])

  if (loading)
    return <div className="text-center py-2">Loading stations…</div>

  if (error)
    return (
      <div className="text-center py-2 text-red-600">
        {error}
      </div>
    )

  return (
    <div className="text-center">
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-[300px] mx-auto">
          <SelectValue placeholder="Select Station" />
        </SelectTrigger>
        <SelectContent className="bg-white dark:bg-gray-800 ">
          {stations.map((s) => (
            <SelectItem key={s.station_id} value={s.station_id} className="
                px-3 py-2 
                hover:bg-gray-100 dark:hover:bg-gray-700 
                hover:text-gray-900 dark:hover:text-white 
                cursor-pointer
              ">
              {s.station_name} ({s.state_outlet})
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
