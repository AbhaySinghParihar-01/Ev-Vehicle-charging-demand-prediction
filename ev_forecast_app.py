# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Load Trained Model ===
model = joblib.load("forecasting_ev_model.pkl")

# === Custom Styling ===
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #c2d3f2, #7f848a);
        }
    </style>
""", unsafe_allow_html=True)

# === Page Title & Header ===
st.markdown("""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: #FFFFFF;'>
        🔮 EV Adoption Forecaster - Washington State Counties
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; font-size: 22px; margin-bottom: 20px; color: #FFFFFF;'>
        Predict EV adoption trends for the next 3 years by selecting a county.
    </div>
""", unsafe_allow_html=True)

st.image("ev-car-factory.jpg", use_container_width=True)

# === Load Dataset ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# === County Selector ===
counties = sorted(df['County'].dropna().unique())
selected_county = st.selectbox("Choose a County", counties)

# === Validate County ===
if selected_county not in df['County'].values:
    st.warning(f"County '{selected_county}' not found in the dataset.")
    st.stop()

county_df = df[df['County'] == selected_county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# === Forecasting Setup ===
historical = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative = list(np.cumsum(historical))
last_month = county_df['months_since_start'].max()
last_date = county_df['Date'].max()

future = []
horizon = 36

for i in range(1, horizon + 1):
    forecast_date = last_date + pd.DateOffset(months=i)
    last_month += 1
    lag1, lag2, lag3 = historical[-1], historical[-2], historical[-3]
    roll_avg = np.mean([lag1, lag2, lag3])
    pct1 = (lag1 - lag2) / lag2 if lag2 else 0
    pct3 = (lag1 - lag3) / lag3 if lag3 else 0
    slope = np.polyfit(range(6), cumulative[-6:], 1)[0] if len(cumulative) >= 6 else 0

    features = pd.DataFrame([{
        'months_since_start': last_month,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_avg,
        'ev_total_pct_change_1': pct1,
        'ev_total_pct_change_3': pct3,
        'ev_growth_slope': slope
    }])

    prediction = model.predict(features)[0]
    future.append({"Date": forecast_date, "Predicted EV Total": round(prediction)})

    historical.append(prediction)
    if len(historical) > 6:
        historical.pop(0)

    cumulative.append(cumulative[-1] + prediction)
    if len(cumulative) > 6:
        cumulative.pop(0)

# === Combine Historical and Forecast Data ===
hist = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
hist['Cumulative EV'] = hist['Electric Vehicle (EV) Total'].cumsum()
hist['Source'] = 'Historical'

forecast = pd.DataFrame(future)
forecast['Cumulative EV'] = forecast['Predicted EV Total'].cumsum() + hist['Cumulative EV'].iloc[-1]
forecast['Source'] = 'Forecast'

combined = pd.concat([hist[['Date', 'Cumulative EV', 'Source']],
                      forecast[['Date', 'Cumulative EV', 'Source']]], ignore_index=True)

# === Plotting ===
st.subheader(f"📊 Cumulative EV Forecast - {selected_county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for source, data in combined.groupby("Source"):
    ax.plot(data['Date'], data['Cumulative EV'], label=source, marker='o')

ax.set_title("Cumulative EV Trend (Next 3 Years)", fontsize=14, color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EV Count", color='white')
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor("#1c1c1c")
ax.tick_params(colors='white')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# === Display Forecast Summary ===
start_total = hist['Cumulative EV'].iloc[-1]
end_total = forecast['Cumulative EV'].iloc[-1]
if start_total:
    change_pct = ((end_total - start_total) / start_total) * 100
    trend = "increase 📈" if change_pct > 0 else "decrease 📉"
    st.success(f"EV adoption in **{selected_county}** is projected to show a **{trend} of {change_pct:.2f}%** in 3 years.")
else:
    st.warning("Insufficient historical data to calculate growth.")

# === Compare Multiple Counties ===
st.markdown("---")
st.header("Compare Forecasts for Multiple Counties")
selected_multi = st.multiselect("Select up to 3 Counties", counties, max_selections=3)

if selected_multi:
    combined_multi = []
    for cty in selected_multi:
        cty_df = df[df['County'] == cty].sort_values("Date")
        cty_code = cty_df['county_encoded'].iloc[0]

        h_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
        c_ev = list(np.cumsum(h_ev))
        mss = cty_df['months_since_start'].max()
        l_date = cty_df['Date'].max()

        future_cty = []
        for i in range(1, horizon + 1):
            forecast_date = l_date + pd.DateOffset(months=i)
            mss += 1
            l1, l2, l3 = h_ev[-1], h_ev[-2], h_ev[-3]
            r_avg = np.mean([l1, l2, l3])
            p1 = (l1 - l2) / l2 if l2 else 0
            p3 = (l1 - l3) / l3 if l3 else 0
            slope = np.polyfit(range(6), c_ev[-6:], 1)[0] if len(c_ev) >= 6 else 0

            row = pd.DataFrame([{
                'months_since_start': mss,
                'county_encoded': cty_code,
                'ev_total_lag1': l1,
                'ev_total_lag2': l2,
                'ev_total_lag3': l3,
                'ev_total_roll_mean_3': r_avg,
                'ev_total_pct_change_1': p1,
                'ev_total_pct_change_3': p3,
                'ev_growth_slope': slope
            }])
            pred = model.predict(row)[0]
            future_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

            h_ev.append(pred)
            if len(h_ev) > 6: h_ev.pop(0)
            c_ev.append(c_ev[-1] + pred)
            if len(c_ev) > 6: c_ev.pop(0)

        hist_cty = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cty['Cumulative EV'] = hist_cty['Electric Vehicle (EV) Total'].cumsum()
        forecast_cty = pd.DataFrame(future_cty)
        forecast_cty['Cumulative EV'] = forecast_cty['Predicted EV Total'].cumsum() + hist_cty['Cumulative EV'].iloc[-1]

        total_cty = pd.concat([hist_cty[['Date', 'Cumulative EV']], forecast_cty[['Date', 'Cumulative EV']]], ignore_index=True)
        total_cty['County'] = cty
        combined_multi.append(total_cty)

    result = pd.concat(combined_multi, ignore_index=True)

    # Plot multi-county comparison
    st.subheader("📁 Multi-County Forecast Comparison")
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    for cty, grp in result.groupby("County"):
        ax2.plot(grp['Date'], grp['Cumulative EV'], label=cty, marker='o')
    ax2.set_facecolor("#1c1c1c")
    fig2.patch.set_facecolor("#1c1c1c")
    ax2.tick_params(colors='white')
    ax2.set_title("Cumulative EV Forecasts (Next 3 Years)", fontsize=16, color='white')
    ax2.set_xlabel("Date", color='white')
    ax2.set_ylabel("Cumulative EV Count", color='white')
    ax2.legend(title="County")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Show growth %
    growths = []
    for cty in selected_multi:
        temp = result[result['County'] == cty].reset_index(drop=True)
        base = temp['Cumulative EV'].iloc[-horizon - 1]
        final = temp['Cumulative EV'].iloc[-1]
        if base:
            pct = ((final - base) / base) * 100
            growths.append(f"{cty}: {pct:.2f}%")
        else:
            growths.append(f"{cty}: N/A")

    st.success("Forecasted EV growth: " + " | ".join(growths))

# === Footer ===
st.markdown("Prepared by **Abhay Singh Parihar** for AICTE Internship Cycle 2")
