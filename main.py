# pip install streamlit fbprophet yfinance plotly
import streamlit as st

import datetime as dt
from datetime import datetime
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from plotly import graph_objs as go
import pandas as pd

st.set_page_config(layout="wide")
st.title("Time Series Forecasting Application")
st.write("This was inspired by [PythonEngineer's Youtube Video] (https://www.youtube.com/watch?v=0E_31WqVzCY)")
st.text("Data Sourced from Yahoo")

START = "1993-01-01"
TODAY = dt.date.today().strftime("%Y-%m-%d")

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

@st.cache
def load_data(ticker, start_date=START): 
    data = yf.download(ticker, start_date, TODAY)
    data.reset_index(inplace=True)
    return data

# Prepare the layout to have two containers because Streamlit is run in order
plot_output1 = st.beta_container()
plot_output2 = st.beta_container()

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close"))
    fig.layout.update(
        title_text="Raw Data", title_x = 0.5, xaxis_rangeslider_visible=True
    )
    plot_output1.plotly_chart(fig, use_container_width=True)

#Create the menu items on left Sidebar

selected_ticker = st.sidebar.selectbox("Select dataset for prediction", assets)
start_date = st.sidebar.text_input("Type a startdate for your date range YYYY-MM-DD Format", START)
data = load_data(selected_ticker, start_date)

st.sidebar.write("Select Prediction Range")
n_months = st.sidebar.slider("Months of prediction:", 1, 48)
days = n_months * 30
# Now enable the user to fine tune the hyperparameters
# TODO: need to add tooltips explaining these
st.sidebar.write("Set Up Hyperparameters [Read More Here] (https://towardsdatascience.com/time-series-analysis-with-facebook-prophet-how-it-works-and-how-to-use-it-f15ecf2c0e3a)")
cp_range = st.sidebar.number_input("Pct of data to train on", min_value=0.0, max_value=1.0, value=0.8) 
cp_scale = st.sidebar.number_input("Changepoint Scale", min_value=0.01, max_value=1.0, value=0.05) 
cp_num = st.sidebar.number_input("Number of Changepoints", value=10) 
plot_raw_data()
with plot_output1: st.header("Choose Your Parameters on the Left then Calculate")
# Predict forecast with Prophet.
with st.sidebar.form(key="forecast"):
    submitted = st.form_submit_button("Calculate")
    if submitted: 
        #set up training dataframe
        df_train = data[["Date","Close"]] 

        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet(changepoint_range=cp_range,changepoint_prior_scale=cp_scale, n_changepoints=cp_num)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        
        # Show and plot forecast
        plot_output2.write(f"Forecast plot for {n_months} months")
        fig1 = plot_plotly(m, forecast)
        plot_output2.plotly_chart(fig1,use_container_width=True)
        # a = add_changepoints_to_plot(fig1,m,forecast)
        plot_output2.subheader("Forecast data")
        plot_output2.write(forecast.tail())

        plot_output2.write("Forecast components")
        fig2 = m.plot_components(forecast)
        plot_output2.write(fig2)


# plot_raw_data(col2)

