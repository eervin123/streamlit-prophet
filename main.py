# pip install streamlit fbprophet yfinance plotly
from matplotlib.pyplot import title
from math import sqrt
import streamlit as st
import time
import datetime as dt
from prophet import Prophet
from prophet.plot import plot_plotly, add_changepoints_to_plot
from plotly import graph_objs as go
import pandas as pd
import vectorbt as vbt
import pandas_ta as ta
import numpy as np

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


num_days=180 #constant here held to work with rolling periods for raw data manipulation

@st.cache
def load_data(ticker: str, start_date: str = START, end_date:str=TODAY) -> pd.DataFrame:
    start = start_date+" UTC"
    end = end_date+" UTC"
    try: 
        price_data = vbt.YFData.download(ticker, start=start, end=end)
    except:
        price_data = vbt.YFData.download(ticker, start=start, end="01-01-2022 UTC")
    return price_data.data[ticker]

# Create a list of the various items we may want to calculate and predict along with their common names
item_names = {
    "Close"                     : "Closing Price",
    "rolling_percent_change"    : "Rolling Percent Change",
    "rolling_volatility"        : "Rolling Volatility",
    "rolling_price_change"      : "Rolling Price Change"
}

def get_key_by_value(dictOfElements: dict, valueToFind: str) ->str:
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            return_key=item[0]
    return return_key

def add_rolling_metrics_to_dataframe(price_data: pd.DataFrame, rolling_period: int = num_days ) -> pd.DataFrame:
    new_df=price_data.copy()
    new_df["rolling_percent_change"]=new_df["Close"].pct_change(periods=rolling_period, fill_method="ffill")
    new_df["rolling_volatility"]=new_df["rolling_percent_change"].rolling(rolling_period).std() * (sqrt((365/rolling_period)))
    new_df["rolling_price_change"]=new_df["Close"].rolling(rolling_period).std()
    return new_df


# Prepare the layout to have containers because Streamlit is run in order
plot_output1 = st.container()
calc_container = st.container()
plot_output2 = st.container()
backtest_container = st.container()

def plot_raw(data: pd.DataFrame, col: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data[col], name= item_names[col]))
    fig.layout.update(autosize=True,
        title_text=f"Raw Data for {item_names[col]} for {selected_ticker}", title_x = 0.5, title_y = .9
    )
    fig.update_layout(margin_b=0, margin_t=0, margin_r=0,margin_l=0, height=300)
    plot_output1.plotly_chart(fig, use_container_width=True)
    
@st.cache
def run_model(data: pd.DataFrame):
    df_train = data[[column_name]]
    df_train.index = df_train.index.tz_localize(None) # Prophet doesn't have timezone
    df_train.reset_index(inplace=True) # Need to reset index to use prophet
    df_train = df_train.rename(columns={"Date": "ds", column_name: "y"})
    m = Prophet(changepoint_range=cp_range,changepoint_prior_scale=cp_scale, n_changepoints=int(cp_num))
    m.fit(df_train)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return df_train, m, forecast

def plot_forecast(data: pd.DataFrame, m: Prophet, forecast: pd.DataFrame) -> None:
    # Show and plot forecast
    fig1 = plot_plotly(m, forecast)
    fig1.layout.update(
        title_text=f"Forecasted For {selected_ticker}", 
        title_x = 0.5, 
        title_y = .9
        )
    fig1.update_layout(margin_b=0, margin_t=0, margin_r=0,margin_l=0)
    plot_output2.plotly_chart(fig1,use_container_width=True)
    
#Create the menu items on left Sidebar
selected_ticker = st.sidebar.selectbox("Select dataset for prediction", assets)
start_date = st.sidebar.text_input("Type a startdate for your date range YYYY-MM-DD Format Note, this will be UTC Time", START)
end_date = st.sidebar.text_input("Type a enddate for your date range YYYY-MM-DD Format Note this will be UTC Time", TODAY)
data = load_data(selected_ticker, start_date, end_date)
# data = add_rolling_metrics_to_dataframe(data, num_days) # I think I will remove this for now

st.sidebar.write("Select Prediction Range")
days = st.sidebar.slider("Days of prediction:", 0, 730, value=30, step=5)

with st.sidebar:
    my_expander = st.expander(label="Tune Hyperparameters?", expanded=True)
    with my_expander:
        st.write("Set Up Hyperparameters")
        st.write("Try the following:")
        st.write("Pct of data to train on=1.0")
        st.write("Changepoint Scale=0.10 works great for BTC, or 0.01 works great for SPY")
        st.write("Number of Changepoints=25")
        cp_range    = st.number_input("Pct of data to train on", min_value=0.0, max_value=1.0, value=1.0) 
        cp_scale    = st.number_input("Changepoint Scale", min_value=0.01, max_value=1.0, value=0.20) 
        cp_num      = st.number_input("Number of Changepoints", value=25) 

# Adding functionality for Volatility Prediction

with plot_output1:
    # I'll add the radio button below to enable them to predict other factors like volatility etc.
#     prediction_statistic = st.radio("What would you like to analyze and predict? ", item_names.values())
#     column_name = get_key_by_value(item_names, prediction_statistic)
    column_name = "Close" # Comment this out when we add the radio button above
    plot_raw(data,column_name)   
    

# with plot_output1: st.header("Choose Your Parameters on the Left then Calculate")
# Predict forecast with Prophet.
with calc_container:
    st.header("Prediction Forecast and a simple Backtest")
    st.write("After Prediction Forecast is made, you can run a backtest based on the prediction model. \
             The simplified backtest will purchase the asset when the price is below the predicted price \
             and vice versa, sell the asset when the price is above the predicted price.")
    with st.form(key="calc"):
        submitted = st.form_submit_button("Calculate Prediction Forecast")
        if submitted: 
            st.spinner()
            start = time.time()
            with st.spinner(text=f"Calculating the forecast for {days} days"):
                df_train, m, forecast = run_model(data)
                plot_forecast(df_train, m, forecast)
                runtime = time.time() - start
                st.success(f"Forecast Completed in {runtime} seconds")
                plot_output2.subheader("Forecast data")
                plot_output2.write(forecast.tail())
                # plot_output2.write("Forecast components")
                # fig2 = m.plot_components(forecast)
                # plot_output2.write(fig2)


with backtest_container:
    with st.form(key="calc_backtest"):
        backtest_submitted = st.form_submit_button("Run a Backtest based on Prediction Forecast")
        if backtest_submitted: 
            st.spinner()
            start = time.time()
            with st.spinner(text=f"Calculating backtest and plotting trades."):
                df_train, m, forecast = run_model(data)
                plot_forecast(df_train, m, forecast)
                st.header("Backtest based on prediction Model")
                st.subheader("Simple Backtest Strategy.")
                st.write("This strategy will buy when the price is above the predicted changepoint and sell when the price is below the predicted changepoint.")
                st.write("The following summary shows exit trades and the overall performance of the strategy")
                
                # Create a backtested Strategy
                power = 0 # Use this as a multiplier/divisor for the forecasted values
                backtest = pd.merge(forecast[["ds","yhat"]], df_train[["ds","y"]], on="ds")
                backtest.set_index("ds", inplace=True)
                entries = backtest["y"].vbt.crossed_below(backtest["yhat"]*(1+power)) # When price is below forecasted price
                exits = backtest["y"].vbt.crossed_above(backtest["yhat"]*(1-power)) # When price is above forecasted price
                
                # Run the backtest and print plot the resusts
                pf = vbt.Portfolio.from_signals(backtest["y"], entries, exits)
                benchmark = backtest["y"].vbt.to_returns() # Create a benchmark for the statistics framework
                fig3 = pf.plot() # Plot the results
                backtest_container.plotly_chart(fig3)
                st.write(f"\$100 invested in the strategy would have generated: \${pf.total_profit():,.2f} in total profit.")
                st.write(f"The strategy would have generated: {pf.total_return()*100:,.2f}% in total return.")
                st.write(f"The benchmark return over that same time period would have been: {pf.total_benchmark_return()*100:,.2f}%.")
                # Print the stats. TODO: come back to this later to clean it up
                backtest_container.header("Backtest Statistics")
                for k,v in pf.stats(settings=dict(benchmark_rets=benchmark)).items():
                    if k.endswith("[%]"):
                        backtest_container.write(f"{k}: {v:.2f}%")
                    elif type(v) == np.float64:
                        backtest_container.write(f"{k}: {v:.2f}")
                    elif type(v) == np.int64:
                        backtest_container.write(f"{k}: {v}")
                    else:
                        backtest_container.write(f"{k}" + ": " + str(v))

                    


st.sidebar.write("Inspired by [PythonEngineer] (https://www.youtube.com/watch?v=0E_31WqVzCY)")
st.sidebar.write("Author: Eric Ervin, Blockforce Capital & Onramp Invest")