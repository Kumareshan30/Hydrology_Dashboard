# main.py
from typing import Optional, List

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import json
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pandas import DateOffset
from pydantic import BaseModel

# For seasonal decomposition and ARIMA forecasting:
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pymannkendall as mk

# For feature importance
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim

import plotly.express as px
from plotly.subplots import make_subplots

from statsmodels.tsa.arima.model import ARIMA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, leaves_list

app = FastAPI()

# Allow CORS for your Next.js frontend (assumed to run on http://localhost:3000)
origins = ["http://localhost:3000","https://hydrology-dashboard.vercel.app/"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


state_values = pd.read_csv(
    "https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CAMELS_AUS_Attributes.Indices_MasterTable.csv",
    usecols=["station_id", "station_name", "state_outlet"])

meta_states = state_values.set_index("station_id")


class Station(BaseModel):
    station_id: str
    station_name: str
    state_outlet: str


@app.get("/stations", response_model=List[Station])
async def list_stations():
    """
    Return a list of all stations with their human‐readable name and state for the dropdown.
    """
    try:
        # only read the three columns we care about, cast to str
        df = pd.read_csv(
            "https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CAMELS_AUS_Attributes.Indices_MasterTable.csv",
            usecols=["station_id", "station_name", "state_outlet"],
            dtype=str,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Could not load stations: {e}"
        )

    # drop any accidental duplicates (by station_id) and sort by station_name
    df = (
        df.drop_duplicates(subset="station_id")
        .sort_values("station_name")
        .reset_index(drop=True)
    )

    # turn into a list of dicts
    stations = df.to_dict(orient="records")
    return stations


@app.get("/soil/hierarchical/states")
async def soil_hierarchical_states():
    """Return list of all unique states for the dropdown."""
    states = sorted(meta_states["state_outlet"].dropna().unique().tolist())
    return states


###########################################
# 1. STREAMFLOW ANALYSIS (Existing Endpoints)
###########################################
def load_streamflow_data(station_id: str):
    """
    Loads a wide-format streamflow_MLd.csv file (with columns year, month, day and station IDs),
    creates a datetime column, selects the given station, replaces missing values (-99.99) with NaN,
    and resamples to monthly averages.
    """
    try:
        df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/streamflow_MLd.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error reading streamflow CSV: " + str(e))

    # Create date from year, month, and day
    try:
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error creating date column: " + str(e))

    if station_id not in df.columns:
        raise HTTPException(status_code=404, detail=f"Station id '{station_id}' not found in streamflow data.")

    catchment_df = df[["date", station_id]].copy()
    catchment_df.rename(columns={station_id: "streamflow"}, inplace=True)
    catchment_df["streamflow"].replace(-99.99, np.nan, inplace=True)
    catchment_df.dropna(subset=["streamflow"], inplace=True)
    catchment_df.set_index("date", inplace=True)
    monthly_flow = catchment_df.resample("ME").mean()
    monthly_flow.dropna(inplace=True)
    return monthly_flow


@app.get("/monthly_flow_plot")
async def monthly_flow_plot(station_id: str = Query(..., description="Station ID for streamflow catchment")):
    monthly_flow = load_streamflow_data(station_id)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_flow.index,
        y=monthly_flow["streamflow"],
        mode="lines+markers",
        name="Streamflow"
    ))
    fig.update_layout(
        title="Monthly Average Streamflow",
        xaxis_title="Date",
        yaxis_title="Streamflow (ML/day)",
        template="plotly_white"
    )
    return json.loads(fig.to_json())


@app.get("/trend_line_plot")
async def trend_line_plot(station_id: str = Query(..., description="Station ID for streamflow catchment")):
    monthly_flow = load_streamflow_data(station_id)
    X = np.array([dt.toordinal() for dt in monthly_flow.index]).reshape(-1, 1)
    y = monthly_flow["streamflow"].values
    reg = LinearRegression().fit(X, y)
    trend = reg.predict(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_flow.index,
        y=monthly_flow["streamflow"],
        mode="markers",
        name="Observed"
    ))
    fig.add_trace(go.Scatter(
        x=monthly_flow.index,
        y=trend,
        mode="lines",
        name="Trend Line"
    ))
    fig.update_layout(
        title="Streamflow Trend Line",
        xaxis_title="Date",
        yaxis_title="Streamflow (ML/day)",
        template="plotly_white"
    )
    return json.loads(fig.to_json())


@app.get("/arima_decomposition_plot")
async def arima_decomposition_plot(station_id: str = Query(..., description="Station ID for streamflow catchment")):
    monthly_flow = load_streamflow_data(station_id)
    try:
        decomposition = seasonal_decompose(monthly_flow["streamflow"], model="additive", period=12)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error in seasonal decomposition: " + str(e))

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
    )
    fig.add_trace(go.Scatter(x=monthly_flow.index, y=decomposition.observed, mode="lines", name="Observed"), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=monthly_flow.index, y=decomposition.trend, mode="lines", name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=monthly_flow.index, y=decomposition.seasonal, mode="lines", name="Seasonal"), row=3,
                  col=1)
    fig.add_trace(go.Scatter(x=monthly_flow.index, y=decomposition.resid, mode="lines", name="Residual"), row=4, col=1)
    fig.update_layout(
        height=800,
        title="ARIMA Decomposition",
        template="plotly_white"
    )
    return json.loads(fig.to_json())


# @app.get("/arima_forecast_plot")
# async def arima_forecast_plot(
#         station_id: str = Query(..., description="Station ID for streamflow catchment"),
#         steps: int = Query(12, description="Number of forecast months")
# ):
#     monthly_flow = load_streamflow_data(station_id)
#     try:
#         model = SARIMAX(monthly_flow["streamflow"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
#         results = model.fit(disp=False)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Error fitting ARIMA model: " + str(e))
#
#     forecast_obj = results.get_forecast(steps=steps)
#     forecast_mean = forecast_obj.predicted_mean
#     forecast_conf_int = forecast_obj.conf_int()
#
#     last_date = monthly_flow.index[-1]
#     forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq="M")
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=monthly_flow.index,
#         y=monthly_flow["streamflow"],
#         mode="lines",
#         name="Historical"
#     ))
#     fig.add_trace(go.Scatter(
#         x=forecast_index,
#         y=forecast_mean,
#         mode="lines+markers",
#         name="Forecast"
#     ))
#     fig.add_trace(go.Scatter(
#         x=list(forecast_index) + list(forecast_index[::-1]),
#         y=list(forecast_conf_int.iloc[:, 0]) + list(forecast_conf_int.iloc[:, 1])[::-1],
#         fill="toself",
#         fillcolor="rgba(0,100,80,0.2)",
#         line=dict(color="rgba(255,255,255,0)"),
#         hoverinfo="skip",
#         showlegend=True,
#         name="Confidence Interval"
#     ))
#     fig.update_layout(
#         title="ARIMA Forecast",
#         xaxis_title="Date",
#         yaxis_title="Streamflow (ML/day)",
#         template="plotly_white"
#     )
#     return json.loads(fig.to_json())
#
#
# @app.get("/mannkendall")
# async def mannkendall(station_id: str = Query(..., description="Station ID for streamflow catchment")):
#     monthly_flow = load_streamflow_data(station_id)
#     try:
#         result = mk.seasonal_test(monthly_flow["streamflow"], period=12)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Error in Mann-Kendall test: " + str(e))
#
#     result_dict = {
#         "trend": result.trend,
#         "p_value": result.p,
#         "test_statistic": result.s,
#         "z": result.z
#     }
#     return result_dict

@app.get("/arima_forecast_plot")
async def arima_forecast_plot(
    station_id: str = Query(..., description="Station ID"),
    steps: int = Query(12, description="Number of forecast months"),
):
    # load full record
    monthly_flow = load_streamflow_data(station_id)
    if monthly_flow.empty:
        raise HTTPException(404, f"No data for station {station_id}")

    # fit ARIMA
    try:
        model = SARIMAX(
            monthly_flow["streamflow"],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
        )
        results = model.fit(disp=False)
    except Exception as e:
        raise HTTPException(500, f"ARIMA model error: {e}")

    # forecast
    fc = results.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int()

    # build forecast index using the recommended "ME" freq
    last = monthly_flow.index[-1]
    idx = pd.date_range(
        start=last + pd.DateOffset(months=1),
        periods=steps,
        freq="ME"  # month-end
    )

    # plot only the last 5 years of history
    hist_start = last - pd.DateOffset(years=10)
    hist_flow = monthly_flow.loc[hist_start:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_flow.index,
        y=hist_flow["streamflow"],
        mode="lines",
        name="Observed (last 5 yrs)"
    ))
    fig.add_trace(go.Scatter(
        x=idx,
        y=mean,
        mode="lines+markers",
        name="Forecast"
    ))
    fig.add_trace(go.Scatter(
        x=list(idx) + list(idx[::-1]),
        y=list(ci.iloc[:, 0]) + list(ci.iloc[:, 1])[::-1],
        fill="toself",
        fillcolor="rgba(0,100,80,0.2)",
        line=dict(color="rgba(0,0,0,0)"),  # fully transparent
        hoverinfo="skip",
        showlegend=True,
        name="95% CI"
    ))
    fig.update_layout(
        title="Streamflow Forecast using ARIMA Timeseries Forecast Model",
        xaxis_title="Date",
        yaxis_title="Streamflow (ML/day)",
        template="plotly_white"
    )
    plot_json = json.loads(fig.to_json())

    # Mann–Kendall on the full series
    try:
        mk_res = mk.seasonal_test(monthly_flow["streamflow"], period=12)
    except Exception as e:
        raise HTTPException(500, f"Mann–Kendall error: {e}")

    mk_dict = {
        "trend": mk_res.trend,
        "p_value": mk_res.p,
        "test_statistic": mk_res.s,
        "z": mk_res.z,
        "interpretation": (
            f"The Mann–Kendall test indicates a {mk_res.trend} trend "
            f"(Z = {mk_res.z:.2f}, p = {mk_res.p:.3f})."
        ),
    }

    return {
        "plot": plot_json,
        "mannkendall": mk_dict,
    }

###########################################
# 2. FEATURE IMPORTANCE ANALYSIS
###########################################
def load_attributes_data():
    """
    Load catchment attributes and hydrological signature from the master table.
    Assumes the CSV 'CAMELS_AUS_Attributes&Indices_MasterTable.csv' has a column 'station_id'
    and a target variable 'sig_mag_Q_mean'. All other columns (except station_id and target)
    are treated as predictors.
    """
    try:
        df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CAMELS_AUS_Attributes.Indices_MasterTable.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error reading attributes CSV: " + str(e))
    return df


@app.get("/feature_importance_rf")
async def feature_importance_rf():
    """
    Trains a Random Forest model to predict the hydrological signature (sig_mag_Q_mean)
    and returns a Plotly bar chart (JSON) of the top 10 feature importances.
    """
    df = load_attributes_data()
    target = "sig_mag_Q_mean"
    if target not in df.columns:
        raise HTTPException(status_code=500, detail=f"Target '{target}' not found in attributes data.")

    # Drop station_id and target, then select only numeric columns as predictors.
    X = df.drop(columns=["station_id", target]).select_dtypes(include=[np.number])
    y = df[target]

    # Fill missing values with the median.
    X.fillna(X.median(), inplace=True)

    # Split data into training and test sets (only training used here).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestRegressor.
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Compute predictions and R² score.
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = r2_score(y_test, y_pred_rf)
    print("Random Forest R^2 Accuracy:", accuracy_rf)

    # Get the top 10 feature importances.
    importance_rf = pd.Series(rf.feature_importances_, index=X.columns)
    importance_rf = importance_rf.sort_values(ascending=False).head(10)

    # Create a horizontal bar chart using Plotly.
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance_rf.values[::-1],
        y=importance_rf.index[::-1],
        orientation="h",
        marker_color="dodgerblue"
    ))
    fig.update_layout(
        title="Top 10 Random Forest Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white"
    )
    return json.loads(fig.to_json())


@app.get("/feature_importance_xgb")
async def feature_importance_xgb():
    """
    Trains an XGBoost model to predict the hydrological signature (sig_mag_Q_mean)
    and returns a Plotly bar chart (JSON) of the top 10 feature importances.
    """
    df = load_attributes_data()
    target = "sig_mag_Q_mean"
    if target not in df.columns:
        raise HTTPException(status_code=500, detail=f"Target '{target}' not found in attributes data.")

    # Drop station_id and target, and select only numeric predictors.
    X = df.drop(columns=["station_id", target]).select_dtypes(include=[np.number])
    y = df[target]
    X.fillna(X.median(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = r2_score(y_test, y_pred_xgb)
    print("XGBoost R^2 Accuracy:", accuracy_xgb)

    importance_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns)
    importance_xgb = importance_xgb.sort_values(ascending=False).head(10)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance_xgb.values[::-1],
        y=importance_xgb.index[::-1],
        orientation="h",
        marker_color="darkorange"
    ))
    fig.update_layout(
        title="Top 10 XGBoost Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white"
    )
    return json.loads(fig.to_json())


###########################################
# 3. ANOMALY & OUTLIER DETECTION
###########################################
def load_hydro_data(station_id: str):
    """
    Loads streamflow, precipitation, and temperature data for a given station.
    Assumes:
      - streamflow_MLd.csv: columns include 'station_id', 'date', 'streamflow'
      - precipitation_agcd.csv: columns include 'station_id', 'date', 'precipitation'
      - tmax_agcd.csv: columns include 'station_id', 'date', 'tmax'
    Merges them on 'station_id' and 'date' and drops missing data.
    """
    try:
        streamflow_df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/streamflow_MLd.csv", parse_dates=["date"])
        precip_df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/precipitation_AGCD.csv", parse_dates=["date"])
        tmax_df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/tmax_AGCD.csv", parse_dates=["date"])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error reading hydrological CSVs: " + str(e))

    # Filter for the selected station_id in each DataFrame
    streamflow_df = streamflow_df[streamflow_df["station_id"] == station_id]
    precip_df = precip_df[precip_df["station_id"] == station_id]
    tmax_df = tmax_df[tmax_df["station_id"] == station_id]

    # Merge dataframes on 'station_id' and 'date'
    try:
        df = pd.merge(streamflow_df, precip_df, on=["station_id", "date"], how="inner")
        df = pd.merge(df, tmax_df, on=["station_id", "date"], how="inner")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error merging hydrological data: " + str(e))

    df.dropna(inplace=True)
    return df


@app.get("/anomaly_detection")
async def anomaly_detection(station_id: str = Query(..., description="Station ID for hydrological anomaly detection")):
    """
    Merges streamflow, precipitation, and temperature data, aggregates monthly,
    and applies Isolation Forest to detect anomalies in streamflow.
    Returns a Plotly figure with the streamflow time series and anomalies marked.
    """
    df = load_hydro_data(station_id)
    # Assume columns: 'streamflow', 'precipitation', 'tmax'
    df.set_index("date", inplace=True)
    df_monthly = df.resample("M").agg({
        "streamflow": "mean",
        "precipitation": "sum",
        "tmax": "mean"
    }).dropna().reset_index()

    # Apply Isolation Forest on selected features
    try:
        iso = IsolationForest(contamination=0.05, random_state=42)
        features = df_monthly[["streamflow", "precipitation", "tmax"]]
        df_monthly["anomaly"] = iso.fit_predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error in anomaly detection: " + str(e))

    # Select anomalies where IsolationForest returns -1
    anomalies = df_monthly[df_monthly["anomaly"] == -1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_monthly["date"],
        y=df_monthly["streamflow"],
        mode="lines",
        name="Streamflow"
    ))
    fig.add_trace(go.Scatter(
        x=anomalies["date"],
        y=anomalies["streamflow"],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Anomalies"
    ))
    fig.update_layout(
        title="Anomaly Detection in Monthly Streamflow",
        xaxis_title="Date",
        yaxis_title="Streamflow (ML/day)",
        template="plotly_white"
    )
    return json.loads(fig.to_json())


@app.get("/predict_xgb")
async def predict_xgb():
    """
    Predicts the hydrological signature (sig_mag_BFI) using an XGBoost model with predictors:
    'distupdamw', 'impound_fac', and 'settlement_fac'. Returns:
      - RMSE and R² metrics,
      - A Plotly scatter plot (actual vs predicted), and
      - A Plotly bar chart of the top 10 feature importances.
    """
    df = load_attributes_data()
    predictors = ['distupdamw', 'impound_fac', 'settlement_fac']
    target = 'sig_mag_BFI'
    if target not in df.columns:
        raise HTTPException(status_code=500, detail=f"Target '{target}' not found.")

    X = df[predictors]
    y = df[target]
    X.fillna(X.median(), inplace=True)
    y.fillna(y.median(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print("XGBoost R^2 Accuracy:", r2)

    # Create Actual vs Predicted scatter plot.
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode="markers",
        name="XGBoost Predictions",
        marker=dict(color="green")
    ))
    scatter_fig.add_trace(go.Scatter(
        x=y_test,
        y=y_test,
        mode="lines",
        name="Ideal",
        line=dict(color="red", dash="dash")
    ))
    scatter_fig.update_layout(
        title="XGBoost: Actual vs Predicted",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        template="plotly_white"
    )

    # Create Feature Importance bar chart.
    importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False).head(10)
    fi_fig = go.Figure()
    fi_fig.add_trace(go.Bar(
        x=importance.values[::-1],
        y=importance.index[::-1],
        orientation="h",
        marker_color="darkorange"
    ))
    fi_fig.update_layout(
        title="XGBoost Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white"
    )

    return {
        "rmse": rmse,
        "r2": r2,
        "predictions_plot": json.loads(scatter_fig.to_json()),
        "feature_importance_plot": json.loads(fi_fig.to_json())
    }


@app.get("/predict_nn")
async def predict_nn():
    """
    Predicts the hydrological signature (sig_mag_BFI) using a PyTorch Neural Network.
    Uses predictors: 'distupdamw', 'impound_fac', 'settlement_fac'.
    Data are standardized and split into training and testing sets.
    Trains the model for 100 epochs, returns:
      - RMSE and R² metrics,
      - A Plotly scatter plot (actual vs. predicted),
      - A Plotly line chart of the training loss history.
    """
    df = load_attributes_data()
    predictors = ['distupdamw', 'impound_fac', 'settlement_fac']
    target = 'sig_mag_BFI'
    if target not in df.columns:
        raise HTTPException(status_code=500, detail=f"Target '{target}' not found.")

    # Prepare dataset
    X = df[predictors]
    y = df[target]
    X.fillna(X.median(), inplace=True)
    y.fillna(y.median(), inplace=True)

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize predictors (important for NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to torch tensors (float32 for NN)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Set up device (CPU for now)
    device = torch.device("cpu")

    # Initialize the model, loss function, and optimizer
    input_dim = X_train_tensor.shape[1]
    model = SimpleNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    batch_size = 32
    train_loss_history = []

    # Simple training loop
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])
        epoch_loss = 0.0

        for i in range(0, X_train_tensor.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_tensor[indices].to(device), y_train_tensor[indices].to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= X_train_tensor.size(0)
        train_loss_history.append(epoch_loss)
        # Optionally, print progress:
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        nn_pred_tensor = model(X_test_tensor)
    nn_pred = nn_pred_tensor.cpu().numpy().flatten()
    rmse_nn = mean_squared_error(y_test, nn_pred, squared=False)
    r2_nn = r2_score(y_test, nn_pred)
    print("Neural Network R^2 Accuracy:", r2_nn)

    # Create Actual vs Predicted scatter plot
    scatter_fig_nn = go.Figure()
    scatter_fig_nn.add_trace(go.Scatter(
        x=y_test,
        y=nn_pred,
        mode="markers",
        name="NN Predictions",
        marker=dict(color="blue")
    ))
    scatter_fig_nn.add_trace(go.Scatter(
        x=y_test,
        y=y_test,
        mode="lines",
        name="Ideal",
        line=dict(color="red", dash="dash")
    ))
    scatter_fig_nn.update_layout(
        title="Neural Network: Actual vs Predicted",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        template="plotly_white"
    )

    # Create training history plot (loss vs epochs)
    history_fig = go.Figure()
    history_fig.add_trace(go.Scatter(
        x=list(range(1, epochs + 1)),
        y=train_loss_history,
        mode="lines",
        name="Training Loss"
    ))
    history_fig.update_layout(
        title="Neural Network Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss (MSE)",
        template="plotly_white"
    )

    return {
        "rmse": rmse_nn,
        "r2": r2_nn,
        "predictions_plot": json.loads(scatter_fig_nn.to_json()),
        "training_history_plot": json.loads(history_fig.to_json())
    }


#
# @app.get("/geospatial_plot")
# def get_geospatial_plot():
#     try:
#         # Load the master attribute table.
#         df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CAMELS_AUS_Attributes.Indices_MasterTable.csv")
#
#         # Ensure required columns are present.
#         required_columns = {"station_id", "lat_outlet", "long_outlet", "state_outlet", "catchment_area"}
#         if not required_columns.issubset(df.columns):
#             missing = required_columns - set(df.columns)
#             raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")
#
#         # 1. Geospatial Scatter Plot.
#         geo_fig = px.scatter_geo(
#             df,
#             lat="lat_outlet",
#             lon="long_outlet",
#             hover_name="station_name",
#             hover_data=["station_id", "station_name", "state_outlet"],
#             size="catchment_area",
#             title="Catchment Outlet Locations in Australia",
#             height=2500,
#             width=2500
#         )
#         # First update the individual geo figure to use the Australia lat/lon ranges.
#         geo_fig.update_geos(
#             scope="world",  # Use the world scope (no built-in "australia")
#             projection_type="natural earth",
#             center={"lat": -25, "lon": 133},
#             lataxis_range=[-40, -20],   # Approximate latitude range for Australia
#             lonaxis_range=[130, 175],   # Approximate longitude range for Australia
#             showcountries=True,
#             countrycolor="black"
#         )
#
#         # 2. Bar Chart: Count of catchments by state.
#         state_freq = df.groupby("state_outlet").size().reset_index(name="count")
#         bar_fig = px.bar(
#             state_freq,
#             x="state_outlet",
#             y="count",
#             title="Number of Catchments by State"
#         )
#         bar_fig.update_layout(
#             xaxis_title="State",
#             yaxis_title="Catchment Count"
#         )
#
#         # 3. Histogram: Distribution of catchment area.
#         hist_fig = px.histogram(
#             df,
#             x="catchment_area",
#             nbins=30,
#             title="Distribution of Catchment Area (km²)"
#         )
#         hist_fig.update_layout(
#             xaxis_title="Catchment Area (km²)",
#             yaxis_title="Frequency"
#         )
#
#         # 4. Optional: Box Plot by Drainage Division (if available).
#         if "drainage_division" in df.columns:
#             box_fig = px.box(
#                 df,
#                 x="drainage_division",
#                 y="catchment_area",
#                 title="Catchment Area by Drainage Division"
#             )
#             box_fig.update_layout(
#                 xaxis_title="Drainage Division",
#                 yaxis_title="Catchment Area (km²)"
#             )
#
#         # Combine subplots.
#         if "drainage_division" in df.columns:
#             fig = make_subplots(
#                 rows=4, cols=1,
#                 specs=[
#                     [{"type": "scattergeo"}],
#                     [{"type": "xy"}],
#                     [{"type": "xy"}],
#                     [{"type": "xy"}]
#                 ],
#                 subplot_titles=[
#                     "Catchment Outlet Locations",
#                     "Catchment Count by State",
#                     "Distribution of Catchment Area",
#                     "Catchment Area by Drainage Division"
#                 ],
#                 vertical_spacing=0.1,
#                 row_heights=[0.4, 0.2, 0.2, 0.2]
#             )
#             for trace in geo_fig.data:
#                 fig.add_trace(trace, row=1, col=1)
#             for trace in bar_fig.data:
#                 fig.add_trace(trace, row=2, col=1)
#             for trace in hist_fig.data:
#                 fig.add_trace(trace, row=3, col=1)
#             for trace in box_fig.data:
#                 fig.add_trace(trace, row=4, col=1)
#             fig.update_layout(height=1600, title_text="Geospatial Analysis of Catchments in Australia")
#             # Update the geo subplot (row 1, col=1) with the Australia view:
#             fig.update_geos(
#                 scope="world",
#                 projection_type="natural earth",
#                 center={"lat": -25, "lon": 133},
#                 lataxis_range=[-52, -15],   # Approximate latitude range for Australia
#                 lonaxis_range=[125, 175],
#                 showcountries=True,
#                 countrycolor="black",
#             )
#         else:
#             fig = make_subplots(
#                 rows=4, cols=1,
#                 specs=[
#                     [{"type": "scattergeo"}],
#                     [{"type": "xy"}],
#                     [{"type": "xy"}]
#                 ],
#                 subplot_titles=[
#                     "Catchment Outlet Locations",
#                     "Catchment Count by State",
#                     "Distribution of Catchment Area"
#                 ],
#                 vertical_spacing=0.1,
#                 row_heights=[0.34, 0.22, 0.22,0.22]
#             )
#             for trace in geo_fig.data:
#                 fig.add_trace(trace, row=1, col=1)
#             for trace in bar_fig.data:
#                 fig.add_trace(trace, row=2, col=1)
#             for trace in hist_fig.data:
#                 fig.add_trace(trace, row=3, col=1)
#             fig.update_layout(height=1800, width=3000, title_text="Geospatial Analysis of Catchments in Australia")
#             fig.update_geos(
#                 scope="world",
#                 projection_type="natural earth",
#                 center={"lat": -25, "lon": 133},
#                 lataxis_range=[-70, 0],   # Approximate latitude range for Australia
#                 lonaxis_range=[120, 190],
#                 showcountries=True,
#                 countrycolor="black",
#                 row=1, col=1
#             )
#
#         # Return a JSON-serializable version of the figure.
#         return json.loads(fig.to_json())
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.get("/geospatial_plot")
def get_geospatial_plot():
    try:
        # Load the master attribute table.
        df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CAMELS_AUS_Attributes.Indices_MasterTable.csv")

        # Ensure required columns are present.
        required_columns = {"station_id", "lat_outlet", "long_outlet", "state_outlet", "catchment_area"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

        # 1. Geospatial Scatter Plot using scatter_mapbox for a modern, interactive map.
        scatter_fig = px.scatter_mapbox(
            df,
            lat="lat_outlet",
            lon="long_outlet",
            hover_name="station_name",
            hover_data=["station_id", "state_outlet"],
            size="catchment_area",
            title="Catchment Outlet Locations in Australia",
            zoom=3,
            center={"lat": -25, "lon": 133},
            height=1200,
            width=1200,
            mapbox_style="open-street-map"  # No Mapbox token required.
        )

        # 2. Bar Chart: Count of catchments by state.
        state_freq = df.groupby("state_outlet").size().reset_index(name="count")
        bar_fig = px.bar(
            state_freq,
            x="state_outlet",
            y="count",
            title="Number of Catchments by State"
        )
        bar_fig.update_layout(
            xaxis_title="State",
            yaxis_title="Catchment Count"
        )

        # Remove the histogram plot ("Distribution of Catchment Area") entirely.

        # 3. Optional: Box Plot by Drainage Division (if available)
        if "drainage_division" in df.columns:
            box_fig = px.box(
                df,
                x="drainage_division",
                y="catchment_area",
                title="Catchment Area by Drainage Division"
            )
            box_fig.update_layout(
                xaxis_title="Drainage Division",
                yaxis_title="Catchment Area (km²)"
            )

            # Combine subplots into three rows: Scatter Map, Bar Chart, Box Plot.
            fig = make_subplots(
                rows=3, cols=1,
                specs=[
                    [{"type": "mapbox"}],
                    [{"type": "xy"}],
                    [{"type": "xy"}]
                ],
                subplot_titles=[
                    "Catchment Outlet Locations",
                    "Catchment Count by State",
                    "Catchment Area by Drainage Division"
                ],
                vertical_spacing=0.1,
                row_heights=[0.4, 0.3, 0.3]
            )
            for trace in scatter_fig.data:
                fig.add_trace(trace, row=1, col=1)
            for trace in bar_fig.data:
                fig.add_trace(trace, row=2, col=1)
            for trace in box_fig.data:
                fig.add_trace(trace, row=3, col=1)
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center={"lat": -25, "lon": 133},
                    zoom=3,
                ),
                height=1600,
                title_text="Geospatial Analysis of Catchments in Australia"
            )
        else:
            # If drainage_division is not available, only combine the scatter map and bar chart.
            fig = make_subplots(
                rows=2, cols=1,
                specs=[
                    [{"type": "mapbox"}],
                    [{"type": "xy"}]
                ],
                subplot_titles=[
                    "Catchment Outlet Locations",
                    "Catchment Count by State"
                ],
                vertical_spacing=0.1,
                row_heights=[0.6, 0.4]
            )
            for trace in scatter_fig.data:
                fig.add_trace(trace, row=1, col=1)
            for trace in bar_fig.data:
                fig.add_trace(trace, row=2, col=1)
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center={"lat": -25, "lon": 133},
                    zoom=3,
                ),
                height=1200,
                width=1200,
                title_text="Geospatial Analysis of Catchments in Australia"
            )

        # Return a JSON-serializable version of the figure.
        return json.loads(fig.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/density_map")
def get_density_map():
    try:
        # Load the master attribute table.
        df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CAMELS_AUS_Attributes.Indices_MasterTable.csv")

        # Make sure the required columns exist.
        required_columns = {"lat_outlet", "long_outlet", "catchment_area"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

        # Create a density map using density_mapbox.
        # Note: Here we set mapbox_style to "open-street-map" to avoid the need for a token.
        density_fig = px.density_mapbox(
            df,
            lat="lat_outlet",
            lon="long_outlet",
            z="catchment_area",  # You can change this to a measure that you’d like to display for density
            radius=10,
            center={"lat": -25, "lon": 133},
            zoom=3,
            mapbox_style="open-street-map",
            title="Density Map of Catchments in Australia",
            height=1000,
            width=1000
        )
        return json.loads(density_fig.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


##############################################
# 1. Hydrometeorological Time Series Analysis
##############################################

@app.get("/hydrometeorology_timeseries")
def get_hydrometeorology_timeseries(
        station_id: str = Query(..., description="Station ID for hydromet analysis")
):
    try:
        # 1) Load the wide-format precipitation file
        df_wide = pd.read_csv(
            "https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/precipitation_AGCD.csv"
        )
        print("here1")
        # The first three columns are 'year', 'month', 'day'.
        # Convert them to a single 'date' column.
        if not {"year", "month", "day"}.issubset(df_wide.columns):
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'year', 'month', and 'day' columns."
            )
        df_wide["date"] = pd.to_datetime(df_wide[["year", "month", "day"]])
        print("here2")
        # 2) Check if station_id is among the columns
        if station_id not in df_wide.columns:
            raise HTTPException(
                status_code=404,
                detail=f"Station ID '{station_id}' not found in precipitation columns."
            )
        print("here3")
        # 3) Create a DataFrame specifically for the chosen station
        #    We'll rename that station column to "precipitation".
        df_station = df_wide[["date", station_id]].copy()
        df_station.rename(columns={station_id: "precipitation"}, inplace=True)
        print("here4")
        # 4) Load the CAMELS master table
        df_master = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CAMELS_AUS_Attributes.Indices_MasterTable.csv")
        print("here5")
        # 5) Attempt to fetch the station's metadata
        row = df_master.loc[df_master["station_id"] == station_id]
        if row.empty:
            # If not found, we can still produce a plot, but station name/state won't be known
            station_name = station_id
            station_state = "Unknown"
        else:
            station_name = row["station_name"].values[0]
            station_state = row["state_outlet"].values[0]
        print("here6")
        # 6) Sort by date (important for line plots) and set as index
        df_station.sort_values("date", inplace=True)
        df_station.set_index("date", inplace=True)
        print("here7")
        # 7) (a) Daily precipitation plot
        fig_daily = px.line(
            df_station.reset_index(),
            x="date",
            y="precipitation",
            title=f"Daily Precipitation: {station_id} - {station_name} ({station_state})",
            labels={"precipitation": "Precipitation (mm)", "date": "Date"},
        )

        # 8) (b) Monthly average precipitation
        df_monthly = (
            df_station["precipitation"]
            .resample("M")  # Resample by end-of-month
            .mean()
            .reset_index()
        )
        fig_monthly = px.line(
            df_monthly,
            x="date",
            y="precipitation",
            title="Monthly Average Precipitation",
            labels={"precipitation": "Precip (mm)", "date": "Date"},
        )

        # 9) (c) Box plot of monthly distribution (across all years)
        df_station["month"] = df_station.index.month  # Extract the month from the daily index
        fig_box = px.box(
            df_station.reset_index(),
            x="month",
            y="precipitation",
            title="Monthly Distribution of Precipitation",
            labels={"month": "Month (1=Jan ... 12=Dec)", "precipitation": "Precip (mm)"},
        )

        # 10) Combine these three figures into a single subplot figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Daily Precipitation",
                "Monthly Average",
                "Monthly Distribution",
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"colspan": 2, "type": "xy"}, None],
            ],
        )

        # (a) Daily precipitation in row=1,col=1
        for trace in fig_daily.data:
            fig.add_trace(trace, row=1, col=1)

        # (b) Monthly average precipitation in row=1,col=2
        for trace in fig_monthly.data:
            fig.add_trace(trace, row=1, col=2)

        # (c) Monthly distribution in row=2,col=1
        for trace in fig_box.data:
            fig.add_trace(trace, row=2, col=1)

        fig.update_layout(
            height=1200,
            title_text=(
                f"Hydrometeorological Time Series Analysis "
                f"({station_id} - {station_name}, {station_state})"
            ),
        )

        # Return the figure in JSON-serializable format
        return json.loads(fig.to_json())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


##############################################
# 2. Climatic Indices Exploration
##############################################

@app.get("/hydrometeorology_indices")
def get_hydrometeorology_indices():
    try:
        # Load the climatic indices file.
        # This file is assumed to contain columns like: station_id, aridity, p_mean, pet_mean, p_seasonality, etc.
        df_indices = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/ClimaticIndices.csv")
        df_indices.rename(columns={"ID": "station_id"}, inplace=True)
        # Load the CAMELS-AUS master attribute table.
        # It should include columns such as: station_id, station_name, state_outlet, etc.
        df_master = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CAMELS_AUS_Attributes.Indices_MasterTable.csv")
        print("here0")

        # Merge the two DataFrames on 'station_id' to add station_name and state_outlet to the indices.
        df = pd.merge(df_indices, df_master[['station_id', 'station_name', 'state_outlet']],
                      on="station_id", how="left")

        print(df)

        # Ensure required columns are present.
        required_columns = {"station_id", "state_outlet", "aridity"}
        print("here1")
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise HTTPException(status_code=400,
                                detail=f"Missing required columns: {missing}")

        # Create a box plot of 'aridity' grouped by state_outlet.
        fig = px.box(
            df,
            x="state_outlet",
            y="aridity",
            title="Distribution of Aridity by State",
            labels={"state_outlet": "State", "aridity": "Aridity (PET / Precipitation)"}
        )
        print("here2")
        fig.update_layout(xaxis_title="State", yaxis_title="Aridity (PET / Precipitation)")
        print("here3")

        # Return the plot as a JSON-serializable object.
        return json.loads(fig.to_json())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


##############################################
# 3. Modeling & Forecasting
##############################################

@app.get("/hydrometeorology_modeling")
def get_hydrometeorology_modeling(
        station_id: str = Query(..., description="Station ID for modeling analysis"),
        start_year: Optional[int] = Query(
            None,
            ge=1900,
            le=2100,
            description="Optional start year for filtering the time series",
        ),
        end_year: Optional[int] = Query(
            None,
            ge=1900,
            le=2100,
            description="Optional end year for filtering the time series",
        ),
):
    try:
        # 1) Load raw daily precipitation
        df_wide = pd.read_csv(
            "https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/precipitation_AGCD.csv"
        )

        # 2) Build datetime
        if not {"year", "month", "day"}.issubset(df_wide.columns):
            raise HTTPException(400, "CSV must contain 'year','month','day'")
        df_wide["date"] = pd.to_datetime(df_wide[["year", "month", "day"]])

        # 3) Validate station
        if station_id not in df_wide.columns:
            raise HTTPException(404, f"Station '{station_id}' not found")

        # 4) Extract station series
        df_station = (
            df_wide[["date", station_id]]
            .rename(columns={station_id: "precipitation"})
            .copy()
        )

        # 5) Apply year‐range filters
        if start_year is not None:
            df_station = df_station[df_station["date"].dt.year >= start_year]
        if end_year is not None:
            df_station = df_station[df_station["date"].dt.year <= end_year]
        if df_station.empty:
            raise HTTPException(404, "No data after applying year‐range filter")

        # 6) Lookup metadata
        df_meta = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CAMELS_AUS_Attributes.Indices_MasterTable.csv")
        m = df_meta.loc[df_meta["station_id"] == station_id]
        if m.empty:
            station_name, station_state = station_id, "Unknown"
        else:
            station_name = m["station_name"].iat[0]
            station_state = m["state_outlet"].iat[0]

        # 7) Resample to monthly means
        df_station.set_index("date", inplace=True)
        monthly = df_station["precipitation"].resample("M").mean()

        # 8) Fit ARIMA(1,1,1) and get in‑sample predictions
        model = ARIMA(monthly, order=(1, 1, 1))
        fit = model.fit()
        pred = fit.predict(
            start=monthly.index[0], end=monthly.index[-1]
        )  # one‑step ahead in sample

        # 9) Prepare DataFrames
        df_act = monthly.reset_index().rename(columns={"precipitation": "Observed"})

        # Only keep forecasts from 1950 onward
        df_pred = pd.DataFrame({
            "date": pred.index,
            "Forecast": pred.values
        })
        df_pred = df_pred[df_pred["date"].dt.year >= 1950]

        # 10) Compose title
        full_start = start_year or int(df_wide["date"].dt.year.min())
        full_end = end_year or int(df_wide["date"].dt.year.max())
        title = (
            f"Monthly Precipitation + ARIMA(1,1,1) In‑Sample Forecast<br>"
            f"{station_id} – {station_name} ({station_state})"
            f"<br>Filtered: {full_start}–{full_end}"
        )

        # 11) Plot both series
        fig = px.line(
            df_act,
            x="date",
            y="Observed",
            title=title,
            labels={"date": "Date", "Observed": "Precipitation (mm)"},
        )
        fig.data[0].name = "Observed"

        fig.add_scatter(
            x=df_pred["date"],
            y=df_pred["Forecast"],
            mode="lines",
            name="Forecast (1950+)",
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Precipitation (mm)",
        )

        return json.loads(fig.to_json())

    except HTTPException:
        # pass through known errors
        raise
    except Exception as e:
        # catch‑all
        raise HTTPException(status_code=500, detail=str(e))


##############################################
# 4. Extreme Value Analysis
##############################################

@app.get("/hydrometeorology_extreme")
def get_hydrometeorology_extreme(
        station_id: str = Query(..., description="Station ID for extreme analysis"),
        threshold_pct: float = Query(
            0.95, ge=0.5, le=0.99, description="Quantile for POT threshold"
        ),
        start_year: Optional[int] = Query(
            None, ge=1900, le=2100, description="Filter: include data ≥ this year"
        ),
        end_year: Optional[int] = Query(
            None, ge=1900, le=2100, description="Filter: include data ≤ this year"
        ),
):
    try:
        # 1) Load raw daily precipitation
        df = pd.read_csv(
            "https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/precipitation_AGCD.csv."
        )

        # 2) Must have year/month/day + station columns
        if not {"year", "month", "day"}.issubset(df.columns):
            raise HTTPException(400, "CSV must have 'year','month','day' columns")
        if station_id not in df.columns:
            raise HTTPException(404, f"Station '{station_id}' not found")

        # 3) Make datetime & select station
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df[["date", station_id]].rename(columns={station_id: "precip"})
        df = df.dropna(subset=["precip"]).sort_values("date")

        # 4) Apply optional year‐range filters
        if start_year is not None:
            df = df[df["date"].dt.year >= start_year]
        if end_year is not None:
            df = df[df["date"].dt.year <= end_year]
        if df.empty:
            raise HTTPException(404, "No data remains after year filters")

        # 5) Lookup station metadata
        meta = (
            pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CAMELS_AUS_Attributes.Indices_MasterTable.csv")
            .query("station_id == @station_id")
        )
        if meta.empty:
            station_name, station_state = station_id, "Unknown"
        else:
            station_name = meta["station_name"].iat[0]
            station_state = meta["state_outlet"].iat[0]

        # 6) POT: find threshold and exceedances
        thresh = df["precip"].quantile(threshold_pct)
        df_pot = df[df["precip"] > thresh]

        # 7) Seasonal block maxima
        season_map = {
            12: "DJF", 1: "DJF", 2: "DJF",
            3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA",
            9: "SON", 10: "SON", 11: "SON"
        }
        df["year"] = df["date"].dt.year
        df["season"] = df["date"].dt.month.map(season_map)
        df_seas = (
            df.groupby(["year", "season"])["precip"]
            .max()
            .reset_index(name="seasonal_max")
        )

        # 8) Build subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                f"POT Exceedances (> {int(100 * threshold_pct)}ᵗʰ pct = {thresh:.1f} mm)",
                "Seasonal Maxima by Year"
            ]
        )

        # Top row: daily & POT
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["precip"], mode="lines", name="Daily"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_pot["date"], y=df_pot["precip"],
                mode="markers", marker=dict(color="red", size=6),
                name="Exceedances"
            ),
            row=1, col=1
        )
        fig.add_hline(y=thresh, line_dash="dash", line_color="firebrick",
                      annotation_text=f"{int(100 * threshold_pct)}ᵗʰ pct",
                      row=1, col=1)

        # Bottom row: seasonal bars
        seasons = ["DJF", "MAM", "JJA", "SON"]
        colors = ["blue", "green", "orange", "purple"]
        for seas, col in zip(seasons, colors):
            dfi = df_seas[df_seas["season"] == seas]
            fig.add_trace(
                go.Bar(
                    x=dfi["year"], y=dfi["seasonal_max"],
                    name=seas, marker_color=col,
                    hovertemplate="Year %{x}<br>" + seas + " max: %{y:.1f} mm"
                ),
                row=2, col=1
            )

        # Layout
        full_start = start_year or int(df["date"].dt.year.min())
        full_end = end_year or int(df["date"].dt.year.max())
        fig.update_layout(
            height=800,
            title=(
                f"Extreme Analysis: {station_id} – {station_name} ({station_state})"
                f"<br>Period: {full_start}–{full_end}"
            ),
            showlegend=True
        )
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=1)
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Max (mm)", row=2, col=1)

        return json.loads(fig.to_json())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


##############################################
# 5. SOIL EXPLORATORY ANALYSIS
##############################################

@app.get("/soil/ksat_boxplot")
async def soil_ksat_boxplot():
    try:
        df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CatchmentAttributes_01_Geology.Soils.csv")
        # ensure station_id is present
        if not {"station_id", "geol_prim", "ksat"}.issubset(df.columns):
            raise HTTPException(400, "Missing 'station_id', 'geol_prim' or 'ksat'")
        # drop NaNs, then merge metadata
        df = df[["station_id", "geol_prim", "ksat"]].dropna()
        df = df.merge(state_values, on="station_id", how="left")

        fig = px.box(
            df,
            x="geol_prim",
            y="ksat",
            title="Boxplot of ksat by Primary Geology",
            labels={"geol_prim": "Primary Geology", "ksat": "ksat (mm h⁻¹)"},
            # show station details on hover
            hover_data=["station_id", "station_name", "state_outlet"],
            points="all"  # show the underlying points too
        )
        return json.loads(fig.to_json())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/soil/clay_sand_scatter")
async def soil_clay_sand_scatter():
    try:
        df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CatchmentAttributes_01_Geology.Soils.csv")
        if not {"station_id", "claya", "sanda", "geol_prim"}.issubset(df.columns):
            raise HTTPException(400, "Missing required columns")
        df = df[["station_id", "claya", "sanda", "geol_prim"]].dropna()
        df = df.merge(state_values, on="station_id", how="left")

        fig = px.scatter(
            df,
            x="claya", y="sanda",
            color="geol_prim",
            title="Clay vs Sand by Primary Geology",
            labels={"claya": "% clay", "sanda": "% sand"},
            hover_data=["station_id", "station_name", "state_outlet"]
        )

        # compute 5% headroom so points don’t sit right on the edge
        max_clay = df["claya"].max() * 1.05
        max_sand = df["sanda"].max() * 1.05

        # force both axes to start at zero
        fig.update_xaxes(range=[0, max_clay])
        fig.update_yaxes(range=[0, max_sand])

        return json.loads(fig.to_json())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/soil/prop_stacked_bar")
async def soil_prop_stacked_bar():
    try:
        df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CatchmentAttributes_01_Geology.Soils.csv")
        prop_cols = ["unconsoldted", "igneous", "silicsed", "carbnatesed",
                     "othersed", "metamorph", "sedvolc", "oldrock"]
        all_cols = ["geol_prim"] + prop_cols
        if not set(all_cols).issubset(df.columns):
            missing = set(all_cols) - set(df.columns)
            raise HTTPException(400, f"Missing columns: {missing}")
        df_avg = df.groupby("geol_prim")[prop_cols].mean().reset_index()
        fig = go.Figure()
        for prop in prop_cols:
            fig.add_trace(go.Bar(
                x=df_avg["geol_prim"], y=df_avg[prop], name=prop
            ))
        fig.update_layout(
            barmode="stack",
            title="Stacked Proportions of Secondary Geology by Primary Type",
            xaxis_title="Primary Geology",
            yaxis_title="Mean Proportion"
        )
        return json.loads(fig.to_json())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/soil/pca_biplot")
async def soil_pca_biplot():
    try:
        df = pd.read_csv("https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CatchmentAttributes_01_Geology.Soils.csv")
        prop_cols = ["unconsoldted", "igneous", "silicsed", "carbnatesed",
                     "othersed", "metamorph", "sedvolc", "oldrock"]
        if not set(prop_cols + ["station_id", "geol_prim"]).issubset(df.columns):
            raise HTTPException(400, "Missing soil or station columns")

        # merge metadata
        df = df[["station_id", "geol_prim"] + prop_cols].fillna(0)
        df = df.merge(state_values, on="station_id", how="left")

        # PCA
        Xs = StandardScaler().fit_transform(df[prop_cols])
        pcs = PCA(n_components=2).fit_transform(Xs)
        df_pca = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        df_pca = pd.concat([df_pca, df[["station_id", "station_name", "state_outlet", "geol_prim"]]], axis=1)

        fig = px.scatter(
            df_pca,
            x="PC1", y="PC2",
            color="geol_prim",
            title="PCA Biplot of Soil‐Proportions",
            labels={"PC1": "PC1", "PC2": "PC2"},
            hover_data=["station_id", "station_name", "state_outlet"]
        )
        return json.loads(fig.to_json())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/soil/kmeans")
async def soil_kmeans(k: int = Query(10, ge=2, le=10)):
    try:
        # 1) Load the exact 12 geology/soil columns
        df = pd.read_csv(
            "https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CatchmentAttributes_01_Geology.Soils.csv"
        )
        features = [
            "unconsoldted",
            "igneous",
            "silicsed",
            "carbnatesed",
            "othersed",
            "metamorph",
            "sedvolc",
            "oldrock",
            "claya",
            "sanda",
            "ksat",
            "solum_thickness",
        ]
        if not set(features).issubset(df.columns):
            raise HTTPException(status_code=400, detail="Missing clustering features")
        X = df[features].fillna(0)

        # 2) Standardize
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        # 3) K‑Means
        km = KMeans(n_clusters=k, random_state=42).fit(Xs)

        # 4) Invert centroids to original units
        centroids_unscaled = scaler.inverse_transform(km.cluster_centers_)
        centroid_df = pd.DataFrame(centroids_unscaled, columns=features)

        # 5) Find which feature each centroid is most above the mean of all centroids
        mean_centroid = centroid_df.mean()
        delta_df = centroid_df.sub(mean_centroid)
        # mapping to human‑readable labels
        label_map = {
            "sanda": "Sand‑Rich",
            "claya": "Clay‑Rich",
            "carbnatesed": "Carbonate‑Dominated",
            "silicsed": "Siliciclastic‑Dominated",
            "igneous": "Igneous‑Rock",
            "metamorph": "Metamorphic‑Rock",
            "othersed": "Other‑Sedimentary",
            "sedvolc": "Sed‑Volcanic",
            "unconsoldted": "Unconsolidated",
            "oldrock": "Old‑Bedrock",
            "ksat": "High‑Ksat",
            "solum_thickness": "Thick‑Solum",
        }
        cluster_label_map = {}
        for idx, row in delta_df.iterrows():
            feat = row.idxmax()
            cluster_label_map[idx] = label_map.get(feat, feat.replace("_", " ").title())

        # 6) Project into PCA(2)
        pcs = PCA(n_components=2).fit_transform(Xs)
        df_plot = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        df_plot["cluster_id"] = km.labels_
        df_plot["cluster_name"] = df_plot["cluster_id"].map(cluster_label_map)

        # 7) Build Plotly figure
        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color="cluster_name",
            title=f"K‑Means (k={k}) Clusters in PCA Space",
            labels={"PC1": "PC1", "PC2": "PC2", "cluster_name": "Cluster Type"},
        )

        return json.loads(fig.to_json())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/soil/hierarchical")
async def soil_hierarchical_heatmap(
        state: str = Query(None, description="Optional state code to filter by (e.g. NSW, QLD)")
):
    try:
        # load raw geology & soils table
        df = pd.read_csv(
            "https://github.com/Kumareshan30/Hydrology_Dashboard/releases/download/v1.0-data/CatchmentAttributes_01_Geology.Soils.csv"
        )

        features = [
            "unconsoldted", "igneous", "silicsed", "carbnatesed",
            "othersed", "metamorph", "sedvolc", "oldrock",
            "claya", "sanda", "ksat", "solum_thickness"
        ]

        label_map = {
            "sanda": "Sand‑Rich",
            "claya": "Clay‑Rich",
            "carbnatesed": "Carbonate‑Dominated",
            "silicsed": "Siliciclastic‑Dominated",
            "igneous": "Igneous‑Rock",
            "metamorph": "Metamorphic‑Rock",
            "othersed": "Other‑Sedimentary",
            "sedvolc": "Sed‑Volcanic",
            "unconsoldted": "Unconsolidated",
            "oldrock": "Old‑Bedrock",
            "ksat": "High‑Ksat",
            "solum_thickness": "Thick‑Solum",
        }

        # ensure we have station_id + all features
        if not set(features + ["station_id"]).issubset(df.columns):
            raise HTTPException(400, "Missing features or station_id")

        # bring in station_name + state_outlet
        df = (
            df.set_index("station_id")
            .join(meta_states[["station_name", "state_outlet"]])
            .reset_index()
        )

        # filter by state if requested
        if state:
            df = df[df["state_outlet"] == state]
            if df.empty:
                raise HTTPException(404, f"No stations found in state {state}")

        # standardize and cluster
        X = df[features].fillna(0).values
        Xs = StandardScaler().fit_transform(X)
        Z = linkage(Xs, method="average")
        order = leaves_list(Z)

        # ordered DataFrame for labels
        ordered_df = df.iloc[order]

        # y-axis labels: "Station Name (STATE)"
        y_labels = [
            f"{row.station_name} ({row.state_outlet})"
            for _, row in ordered_df.iterrows()
        ]

        # x-axis labels via label_map
        x_labels = [label_map[f] for f in features]

        # build heatmap
        heatmap = go.Heatmap(
            z=Xs[order],
            x=x_labels,
            y=y_labels,
            colorscale="Viridis"
        )
        fig = go.Figure(data=heatmap)

        # size dynamically
        n = len(y_labels)
        fig.update_layout(
            title="Hierarchical Clustering Heatmap of Soil Attributes",
            xaxis_title="Attribute",
            yaxis_title="Station – Name (State)",
            height=max(300 + 20 * n, 800),
            margin=dict(l=250, r=50, t=100, b=100)
        )

        return json.loads(fig.to_json())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
