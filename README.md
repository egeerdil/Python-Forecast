# USDTRYForecast

**USD/TRY Exchange Rate Forecasting Model**

A comprehensive machine learning forecasting system for USD/TRY (US Dollar to Turkish Lira) exchange rates using multiple models.

## Features

- **Multiple ML Models**: 
  - Linear Regression
  - Random Forest
  - LSTM (Long Short-Term Memory) Neural Network
  - ARIMA (AutoRegressive Integrated Moving Average)

- **Forecast Horizons**:
  - 3 months (90 days)
  - 6 months (180 days)
  - 1 year (365 days)

- **Historical Data**: Uses 10 years of historical USD/TRY exchange rate data

- **Comprehensive Features**: 
  - Lag features
  - Moving averages
  - Volatility indicators
  - Rate of change
  - RSI (Relative Strength Index)
  - Date features

## Installation

### Option 1: Using the setup script (Recommended)
```bash
./setup.sh
```

### Option 2: Manual installation
1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Activate the virtual environment (if using one):
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the forecasting script:
```bash
python usd_try_forecast.py
```

The script will:
1. Fetch historical USD/TRY data from the last 10 years
2. Preprocess and create features
3. Train multiple machine learning models
4. Generate forecasts for 3, 6, and 12 months
5. Display results and save visualizations

## Output

- **Console Output**: Model performance metrics and forecast summaries
- **Visualization**: `usd_try_forecasts.png` - Charts showing historical data and forecasts for all horizons

## Model Performance

Each model's performance is evaluated using:
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## Notes

- The script uses `yfinance` to fetch real-time data. If the API is unavailable, it will generate synthetic data for demonstration.
- Forecasts are based on historical patterns and should be used for informational purposes only.
- Exchange rates are influenced by many factors not captured in historical data alone.

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

