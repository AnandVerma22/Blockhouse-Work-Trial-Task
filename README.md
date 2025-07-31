
Optimal Order Execution Using Market Impact Models
==================================================
Project Summary
------------------
This project is developed to simulate and optimize stock trading strategies by minimizing slippage and execution cost using different market impact models. It involves reading Level 2 order book data, estimating temporary market impacts, fitting models to the estimated slippage data, and generating optimal execution schedules for different stock tickers.

The project is especially useful in algorithmic trading and electronic execution systems where large orders need to be split into smaller orders to minimize market impact. The models implemented mimic realistic trading behavior by accounting for liquidity availability and order book depth.

Data Overview
----------------
The project operates on a structured directory of CSV files containing order book data. Each directory corresponds to a stock ticker (e.g., SOUN, FROG, CRWV). Each CSV file contains fields such as bid/ask prices and sizes at different levels (L2 data), and timestamps indicating the time of each snapshot.

The columns in the CSVs include:
- ts_event: Timestamp of the snapshot.
- bid_px_00 to bid_px_09: Bid prices from level 0 to level 9.
- ask_px_00 to ask_px_09: Ask prices from level 0 to level 9.
- bid_sz_00 to bid_sz_09: Bid sizes corresponding to bid prices.
- ask_sz_00 to ask_sz_09: Ask sizes corresponding to ask prices.

Objectives and Methodology
-----------------------------
The primary goals of this project include:

1. Loading and Preprocessing Data:
   - Iterates through all ticker directories.
   - Parses timestamp data correctly, accounting for format differences (e.g., for ticker CRWV).
   - Drops rows with missing timestamps.
   - Concatenates all valid CSV dataframes into a master dataframe per ticker.

2. Slippage Calculation:
   - Simulates a trade of given quantity at a given order book snapshot.
   - Calculates the effective execution price and compares it with the mid-price.
   - Measures the difference as the temporary market impact (slippage).

3. Temporary Market Impact Estimation:
   - For each ticker, takes a snapshot of the order book.
   - Simulates buy-side execution for order sizes from 1 to 10,000 shares.
   - Records the per-share slippage for each order size.
   - Plots the resulting impact curve.

4. Model Fitting:
   - Fits three market impact models:
     a. Linear: g(x) = a * x
     b. Square Root: g(x) = a * sqrt(x)
     c. Power Law: g(x) = a * x^b
   - Uses polynomial/logarithmic regression to estimate model parameters.
   - Stores coefficients a and b for further use.

5. Optimal Execution Schedule:
   - Distributes a total order size (e.g., 10,000 shares) over a given time horizon (e.g., 390 minutes).
   - Computes execution weights using each of the fitted models.
   - Compares them with the basic TWAP strategy (equal time-weighted allocation).

6. Visualization:
   - Impact Function Plot: Slippage vs Order Size.
   - Execution Schedule Plot: Shares per Minute vs Time.

Code Structure
-----------------
- OrderBookAnalyzer: Main class that handles all logic.
  - `__init__()`: Initializes object and loads all ticker data.
  - `load_all_data()`: Reads CSV files and stores per-ticker data.
  - `calculate_mid_price()`: Returns mid-point between best bid and ask.
  - `calculate_slippage()`: Estimates slippage based on walking the order book.
  - `estimate_temporary_impact()`: Computes slippage curve for a snapshot.
  - `plot_impact_function()`: Visualizes temporary market impact curve.
  - `fit_impact_model()`: Fits linear, sqrt, and power models to slippage.
  - `optimal_execution()`: Generates and returns optimal execution schedule.

Observations
---------------
- Square root models typically lead to aggressive early execution, anticipating high costs for delayed liquidity consumption.
- Power law models adapt well depending on the estimated exponent `b`, providing flexibility in schedule.
- TWAP provides a baseline but doesn't consider market impact, often suboptimal for large orders.

Sample Execution
-------------------
Sample usage for ticker "SOUN":

```
analyzer = OrderBookAnalyzer("data")
analyzer.plot_impact_function("SOUN")
models = analyzer.fit_impact_model("SOUN")
schedule = analyzer.optimal_execution(S=10000, N=390, ticker="SOUN", model_type="sqrt")
```

Dependencies
---------------
- `pandas`: For data manipulation and CSV loading.
- `numpy`: For numerical operations and model fitting.
- `matplotlib`: For plotting and visualization.

Folder Structure
-------------------
data/
├── SOUN/
│   ├── file1.csv
│   ├── file2.csv
├── FROG/
│   ├── file1.csv
│   ├── file2.csv
├── CRWV/
│   ├── file1.csv
│   ├── file2.csv

Results
----------
Each ticker shows a unique slippage curve and optimal schedule based on its liquidity profile. For instance:
- CRWV showed steep slippage early, favoring early execution.
- SOUN had flatter impact, making TWAP relatively close to optimal.
- FROG had highly nonlinear behavior, showing benefits from power-law scheduling.

Conclusion
-------------
This project demonstrates the importance of understanding temporary market impact for large orders. It shows how different model assumptions lead to different execution strategies, and how these strategies can be adapted per ticker depending on liquidity conditions.

This work is suitable for traders, quant researchers, or developers working in the domain of automated execution and market microstructure.

