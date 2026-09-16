import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import glob
import os


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Ferry Demand Forecasting",
    page_icon="⛴️",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("⛴️ Short-Term Ferry Ticket Demand Forecasting")

st.write(
    "Predictive decision-support dashboard for short-term "
    "Toronto Island ferry ticket demand."
)

st.divider()


# ==========================================
# LOAD MODEL RESULTS
# ==========================================

try:
    results = pd.read_csv("complete_model_comparison.csv")
except FileNotFoundError:
    st.error("complete_model_comparison.csv not found.")
    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Forecast Settings")

horizon = st.sidebar.selectbox(
    "Select Forecast Horizon",
    ["15 Minutes", "30 Minutes", "1 Hour", "2 Hours"]
)


# ==========================================
# FILTER MODEL RESULTS
# ==========================================

horizon_data = results[
    results["Horizon"] == horizon
].copy()


# ==========================================
# BEST MODEL
# ==========================================

best_row = horizon_data.loc[
    horizon_data["MAE"].idxmin()
]

best_model = best_row["Model"]
best_mae = best_row["MAE"]
best_rmse = best_row["RMSE"]
best_mape = best_row["MAPE (%)"]


# ==========================================
# KPI SECTION
# ==========================================

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Best Model",
    best_model
)

col2.metric(
    "MAE",
    f"{best_mae:.2f}"
)

col3.metric(
    "RMSE",
    f"{best_rmse:.2f}"
)

if pd.isna(best_mape):
    col4.metric("MAPE", "N/A")
else:
    col4.metric(
        "MAPE",
        f"{best_mape:.2f}%"
    )


st.divider()


# ==========================================
# MODEL COMPARISON
# ==========================================

st.subheader(
    f"Model Comparison — {horizon}"
)

display_data = horizon_data[
    ["Model", "MAE", "RMSE", "MAPE (%)"]
].copy()

st.dataframe(
    display_data,
    width="stretch",
    hide_index=True
)


# ==========================================
# MAE GRAPH
# ==========================================

st.subheader("MAE Comparison")

fig, ax = plt.subplots()

ax.bar(
    horizon_data["Model"],
    horizon_data["MAE"]
)

ax.set_xlabel("Model")
ax.set_ylabel("MAE")
ax.set_title(
    f"MAE Comparison — {horizon}"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

st.pyplot(fig)


# ==========================================
# RMSE GRAPH
# ==========================================

st.subheader("RMSE Comparison")

fig2, ax2 = plt.subplots()

ax2.bar(
    horizon_data["Model"],
    horizon_data["RMSE"]
)

ax2.set_xlabel("Model")
ax2.set_ylabel("RMSE")
ax2.set_title(
    f"RMSE Comparison — {horizon}"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

st.pyplot(fig2)


# ==========================================
# ARIMA FORECAST SECTION
# ==========================================

st.divider()

st.header("📈 ARIMA Forecast")

st.write(
    "The following section displays the generated ARIMA forecast "
    "and prediction uncertainty."
)


# Find ARIMA forecast file automatically
forecast_files = glob.glob("arima_forecast*.csv")

if len(forecast_files) == 0:

    st.warning(
        "ARIMA forecast file was not found in the project folder."
    )

else:

    forecast_file = forecast_files[0]

    forecast_data = pd.read_csv(forecast_file)
    st.write("Forecast shape:", forecast_data.shape)
    st.write("Forecast columns:", forecast_data.columns.tolist())
    st.write("First rows:", forecast_data.head())
    st.write("Last rows:", forecast_data.tail())
    st.success(
        f"Forecast loaded from: {os.path.basename(forecast_file)}"
    )

    st.write("Forecast data:")

    st.dataframe(
        forecast_data.head(10),
        width="stretch",
        hide_index=True
    )


    # --------------------------------------
    # TRY TO IDENTIFY COLUMNS
    # --------------------------------------

    columns = forecast_data.columns.tolist()

    timestamp_col = None
    actual_col = None
    forecast_col = None
    lower_col = None
    upper_col = None


    for col in columns:

        name = col.lower()

        if (
            "time" in name
            or "date" in name
            or "timestamp" in name
        ):
            timestamp_col = col

        elif (
            "actual" in name
            or "sales" in name
            or "demand" in name
        ):
            if actual_col is None:
                actual_col = col

        if (
            "forecast" in name
            or "pred" in name
        ):
            forecast_col = col

        if (
            "lower" in name
            or "low" in name
        ):
            lower_col = col

        if (
            "upper" in name
            or "high" in name
        ):
            upper_col = col


    # --------------------------------------
    # FORECAST GRAPH
    # --------------------------------------
    
    if forecast_col is not None:

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        

    # Find timestamp column safely
        possible_timestamp_cols = [
            "DateTime",
            "Datetime",
            "datetime",
            "Timestamp",
            "timestamp",
            "Date",
            "date"
        ]

        detected_timestamp_col = next(
            (
                col for col in possible_timestamp_cols
                if col in forecast_data.columns
            ),
        None
        )

    # Use DateTime column if available
        if detected_timestamp_col is not None:

            x_values = pd.to_datetime(
                forecast_data[detected_timestamp_col],
                errors="coerce"
            )

        else:

        # Fallback to row numbers
            x_values = range(len(forecast_data))
        
        
        # Actual values, if available
        if actual_col is not None:
            ax3.plot(
                x_values,
                forecast_data[actual_col],
                marker="o",
                label="Actual Sales"
            )

        # ARIMA forecast
        ax3.plot(
            x_values,
            forecast_data[forecast_col],
            marker="o",
            label="ARIMA Forecast"
        )

        # Prediction interval
        if (
            lower_col is not None
            and upper_col is not None
            and lower_col in forecast_data.columns
            and upper_col in forecast_data.columns
        ):
            ax3.fill_between(
                x_values,
                forecast_data[lower_col],
                forecast_data[upper_col],
                alpha=0.25,
                label="95% Prediction Interval"
            )

        ax3.set_xlabel("Date")
        ax3.set_ylabel("Forecasted Ticket Demand")
        ax3.set_title("ARIMA Ferry Ticket Demand Forecast")

        ax3.legend()
        
        fig3.autofmt_xdate(rotation=45)

        plt.tight_layout()

        st.pyplot(fig3)

    else:
        st.warning(
            "Forecast column could not be identified automatically."
        )
    


# ==========================================
# FORECAST INTERPRETATION
# ==========================================

st.subheader("Forecast Interpretation")

st.write(
    f"For the selected **{horizon}** horizon, "
    f"the current evaluation identifies **{best_model}** "
    f"as the model with the lowest MAE."
)

st.write(
    "Prediction intervals provide an indication of forecast "
    "uncertainty and can help operational users understand "
    "the possible range of future ferry ticket demand."
)


# ==========================================
# OPERATIONAL DECISION SUPPORT
# ==========================================

st.divider()

st.header("🛳️ Operational Decision Support")

st.write(
    "Short-term demand forecasts can support proactive "
    "planning for ferry scheduling, staffing, passenger flow, "
    "and crowd management."
)

st.info(
    "The forecasting system is intended as a decision-support "
    "tool and does not replace operational judgement."
)