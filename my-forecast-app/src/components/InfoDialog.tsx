// components/InfoDialog.tsx
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
  } from "@/components/ui/dialog";
  import { Info } from "lucide-react";
  
  export default function InfoDialog({ title, children }: { title: string; children: React.ReactNode }) {
    return (
      <Dialog>
        <DialogTrigger asChild>
          <button className="ml-2 text-sm text-gray-400 hover:text-black">
            <Info className="w-4 h-4" />
          </button>
        </DialogTrigger>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{title}</DialogTitle>
            <DialogDescription>{children}</DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    );
  }
  