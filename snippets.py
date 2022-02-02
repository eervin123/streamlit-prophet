#%%
import vectorbt as vbt
import yfinance as yf
import datetime as dt
#%%
START = "1993-01-01"
TODAY = dt.date.today().strftime("%Y-%m-%d")
start = START+" UTC"
end = TODAY+" UTC"
spy = vbt.YFData.download("spy", start, end)
btc = vbt.YFData.download("btc-usd", start, end)
spy.data

# %%
# spy.index = spy.index.tz_localize(None)
# btc.index = btc.index.tz_localize(None)
# %%

(spy["Close"].vbt.plot_against(btc["Close"], title="SPY vs. BTC", ylabel="SPY", xlabel="BTC"))
# %%
