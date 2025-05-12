import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils"; // if using class merging

export default function PlotCard({
  title,
  children,
  className,
}: {
  title: React.ReactNode; // <== change from `string` to `React.ReactNode`
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Card className={cn("p-4 shadow-md bg-white border border-gray-200", className)}>
      <h2 className="text-xl font-semibold text-center text-black mb-4">{title}</h2>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

