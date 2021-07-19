# streamlit-prophet

This project is a simple application which enables the user to select a stock, ETF, or crypto then a forecast period in order to run a prediction algorithm using the FBprophet library. The project was originally inspired by `pythonengineer`'s youtube video <https://www.youtube.com/watch?v=0E_31WqVzCY&t=536s>

## Dependencies

fbprophet
streamlit
plotly
yfinance

## Run

To run type `streamlit run main.py`

### TODO

Create several new features enabling the manual selection of a time period for training, creating training on model portfolios built using the ORA tools pages, etc.  

Also would like to set up some training for evaluating optimal rebalance frequencies based on various assets to help optimize portfolio allocations.

One other thing would be to run this on volatility to try to predict volatility going forward. FB Prophet is known for its strengths with seasonality benefits. 
