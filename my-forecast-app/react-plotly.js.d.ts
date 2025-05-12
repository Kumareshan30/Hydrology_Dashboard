// react-plotly.js.d.ts
declare module "react-plotly.js" {
    import * as React from "react";
  
    export interface PlotParams {
      data: any[];
      layout: any;
      config?: any;
      style?: React.CSSProperties;
      className?: string;
    }
  
    const Plot: React.FC<PlotParams>;
    export default Plot;
  }
  