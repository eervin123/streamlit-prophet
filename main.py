# pip install streamlit fbprophet yfinance plotly
from matplotlib.pyplot import title
import streamlit as st
import time
import datetime as dt
from datetime import datetime
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, add_changepoints_to_plot
from plotly import graph_objs as go
import pandas as pd

st.set_page_config(layout="wide")
st.title("Time Series Forecasting For Asset Prices")


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
def load_data(ticker, start_date=START, end_date=TODAY): 
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data

# Prepare the layout to have containers because Streamlit is run in order
plot_output1 = st.container()
calc_container = st.container()
plot_output2 = st.container()

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close"))
    fig.layout.update(autosize=True,
        title_text=f"Raw Price Data For {selected_ticker}", title_x = 0.5, title_y = .9
    )
    fig.update_layout(margin_b=0, margin_t=0, margin_r=0,margin_l=0, height=300)
    plot_output1.plotly_chart(fig, use_container_width=True)

#Create the menu items on left Sidebar

selected_ticker = st.sidebar.selectbox("Select dataset for prediction", assets)
start_date = st.sidebar.text_input("Type a startdate for your date range YYYY-MM-DD Format", START)
end_date = st.sidebar.text_input("Type a enddate for your date range YYYY-MM-DD Format", TODAY)
data = load_data(selected_ticker, start_date, end_date)

st.sidebar.write("Select Prediction Range")
days = st.sidebar.slider("Days of prediction:", 0, 730, value=365, step=5)
# days = n_months * 30
# Now enable the user to fine tune the hyperparameters
# TODO: need to add tooltips explaining these

with st.sidebar:
    my_expander = st.expander(label="Tune Hyperparameters?", expanded=True)
    with my_expander:
        st.write("Set Up Hyperparameters [Read More Here] (https://towardsdatascience.com/time-series-analysis-with-facebook-prophet-how-it-works-and-how-to-use-it-f15ecf2c0e3a)")
        cp_range = st.number_input("Pct of data to train on", min_value=0.0, max_value=1.0, value=1.0) 
        cp_scale = st.number_input("Changepoint Scale", min_value=0.01, max_value=1.0, value=0.20) 
        cp_num = st.number_input("Number of Changepoints", value=10) 

plot_raw_data()
# with plot_output1: st.header("Choose Your Parameters on the Left then Calculate")
# Predict forecast with Prophet.
with calc_container:
    with st.form(key="calc"):
        submitted = st.form_submit_button("Calculate")
        if submitted: 
            #set up training dataframe
            st.spinner()
            start = time.time()
            with st.spinner(text=f"Calculating the forecast for {days} days"):
                df_train = data[["Date","Close"]]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
                m = Prophet(changepoint_range=cp_range,changepoint_prior_scale=cp_scale, n_changepoints=int(cp_num))
                m.fit(df_train)
                future = m.make_future_dataframe(periods=days)
                forecast = m.predict(future)

                # Show and plot forecast
                fig1 = plot_plotly(m, forecast)
                fig1.layout.update(
                    title_text=f"Forecasted Price For {selected_ticker}", title_x = 0.5, title_y = .9
                )
                fig1.update_layout(margin_b=0, margin_t=0, margin_r=0,margin_l=0)
                plot_output2.plotly_chart(fig1,use_container_width=True)
                # a = add_changepoints_to_plot(fig1,m,forecast)
                plot_output2.subheader("Forecast data")
                plot_output2.write(forecast.tail())

                plot_output2.write("Forecast components")
                fig2 = m.plot_components(forecast)
                plot_output2.write(fig2)
                runtime = time.time() - start
                st.success(f"Forecast Completed in {runtime} seconds")

st.sidebar.write("This was inspired by [PythonEngineer's Youtube Video] (https://www.youtube.com/watch?v=0E_31WqVzCY)")
st.sidebar.write("Author: Eric Ervin, Blockforce Capital & Onramp Invest")