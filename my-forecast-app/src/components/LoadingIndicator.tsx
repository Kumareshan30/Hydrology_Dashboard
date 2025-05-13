import { Skeleton } from "@/components/ui/skeleton";

export default function LoadingIndicator({ tab }: { tab: string }) {
    return (
      <div className="space-y-4 max-w-5xl mx-auto px-6">
    <Skeleton className="h-6 w-64 mx-auto" />
    <p className="text-center text-sm text-gray-500">Loading {tab}…</p>
    <Skeleton className="h-[500px] w-full rounded-md" />
  </div>
    );
  }
  