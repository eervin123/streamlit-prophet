# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "1993-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

assets = (
    "SPY",
    "AGG",
    "QQQ",
    "BTC-USD",
    "ETH-USD",
    "MSFT",
    "GOOG",
    "AAPL",
    "IBM",
    "KO",
)
selected_ticker = st.selectbox("Select dataset for prediction", assets)

n_months = st.slider("Months of prediction:", 1, 48)
days = n_months * 30


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Loading data...")
data = load_data(selected_ticker)
data_load_state.text("Loading data... done!")

st.subheader("Raw data")
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close"))
    fig.layout.update(
        title_text="Time Series data with Rangeslider", xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig)


plot_raw_data()

# Predict forecast with Prophet.
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=days)
forecast = m.predict(future)

# Show and plot forecast
st.subheader("Forecast data")
st.write(forecast.tail())

st.write(f"Forecast plot for {n_months} months")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
