"""
USDTRYForecast - USD/TRY Exchange Rate Forecasting Model
========================================================
Uses multiple machine learning models to forecast USD/TRY exchange rates
for 3 months, 6 months, and 1 year horizons.

Project: USDTRYForecast
Author: Machine Learning Forecasting System
"""

PROJECT_NAME = "USDTRYForecast"
VERSION = "1.0.0"

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class USDTRYForecaster:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = MinMaxScaler()
        self.lstm_scaler = MinMaxScaler()
        self.forecasts = {}
        
    def fetch_data(self, years=10):
        """Fetch USD/TRY exchange rate data for the last N years"""
        print(f"Fetching USD/TRY data for the last {years} years...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        try:
            # Fetch USD/TRY data
            ticker = yf.Ticker("USDTRY=X")
            self.data = ticker.history(start=start_date, end=end_date)
            
            if self.data.empty:
                raise ValueError("No data retrieved. Trying alternative method...")
            
            # Save full data before processing
            full_data = self.data.copy()
            full_data.to_csv('usd_try_full_data.csv')
            print(f"✅ Full data saved to 'usd_try_full_data.csv'")
            
            # Use Close price as the main feature
            self.data = self.data[['Close']].copy()
            self.data.columns = ['USDTRY']
            self.data = self.data.dropna()
            
            print(f"Successfully fetched {len(self.data)} days of data")
            print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            print(f"Current USD/TRY rate: {self.data['USDTRY'].iloc[-1]:.4f}")
            
            # Display data summary
            print("\n" + "-"*60)
            print("DATA SUMMARY")
            print("-"*60)
            print(f"Min rate: {self.data['USDTRY'].min():.4f}")
            print(f"Max rate: {self.data['USDTRY'].max():.4f}")
            print(f"Mean rate: {self.data['USDTRY'].mean():.4f}")
            print(f"Std deviation: {self.data['USDTRY'].std():.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Creating synthetic data for demonstration...")
            # Create synthetic data if API fails
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            base_rate = 30.0
            trend = np.linspace(20, 35, len(dates))
            noise = np.random.normal(0, 1, len(dates))
            seasonal = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
            rates = trend + noise + seasonal
            self.data = pd.DataFrame({'USDTRY': rates}, index=dates)
            print(f"Created synthetic data with {len(self.data)} days")
            return False
    
    def create_features(self, df, lookback=30):
        """Create features for machine learning models"""
        df = df.copy()
        
        # Lag features
        for lag in [1, 3, 7, 14, 30]:
            df[f'lag_{lag}'] = df['USDTRY'].shift(lag)
        
        # Moving averages
        df['ma_7'] = df['USDTRY'].rolling(window=7).mean()
        df['ma_30'] = df['USDTRY'].rolling(window=30).mean()
        df['ma_90'] = df['USDTRY'].rolling(window=90).mean()
        
        # Volatility
        df['volatility_7'] = df['USDTRY'].rolling(window=7).std()
        df['volatility_30'] = df['USDTRY'].rolling(window=30).std()
        
        # Rate of change
        df['roc_7'] = df['USDTRY'].pct_change(7)
        df['roc_30'] = df['USDTRY'].pct_change(30)
        
        # Technical indicators
        df['rsi'] = self.calculate_rsi(df['USDTRY'], period=14)
        
        # Date features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        
        return df.dropna()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_lstm_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        scaled_data = self.lstm_scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaled_data
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train Linear Regression model"""
        print("\nTraining Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        self.models['LinearRegression'] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("\nTraining Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        self.models['RandomForest'] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    
    def train_lstm(self, X_train, y_train, X_test, y_test):
        """Train LSTM model"""
        print("\nTraining LSTM model...")
        
        # Reshape for LSTM [samples, time steps, features]
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train with early stopping
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train_lstm, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_lstm, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        train_pred = model.predict(X_train_lstm, verbose=0)
        test_pred = model.predict(X_test_lstm, verbose=0)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        self.models['LSTM'] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    
    def train_arima(self, train_data, test_data):
        """Train ARIMA model"""
        print("\nTraining ARIMA model...")
        try:
            # Find optimal parameters using auto_arima approach (simplified)
            model = ARIMA(train_data, order=(5, 1, 2))
            fitted_model = model.fit()
            
            # Forecast for test period
            forecast = fitted_model.forecast(steps=len(test_data))
            
            test_mae = mean_absolute_error(test_data, forecast)
            test_r2 = r2_score(test_data, forecast)
            
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            
            self.models['ARIMA'] = {
                'model': fitted_model,
                'train_mae': None,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
        except Exception as e:
            print(f"  ARIMA training failed: {e}")
            self.models['ARIMA'] = None
    
    def train_all_models(self):
        """Train all models"""
        print("\n" + "="*60)
        print("PREPARING DATA AND TRAINING MODELS")
        print("="*60)
        
        # Create features
        feature_data = self.create_features(self.data)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(feature_data) * 0.8)
        train_data = feature_data.iloc[:split_idx]
        test_data = feature_data.iloc[split_idx:]
        
        # Prepare features and target
        feature_cols = [col for col in feature_data.columns if col != 'USDTRY']
        X_train = train_data[feature_cols].values
        y_train = train_data['USDTRY'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['USDTRY'].values
        
        # Scale features for tree-based models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        self.train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test)
        self.train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Prepare LSTM data
        lookback = 60
        lstm_data = self.data['USDTRY'].values
        X_lstm, y_lstm, scaled_lstm = self.prepare_lstm_data(self.data[['USDTRY']], lookback)
        
        lstm_split = int(len(X_lstm) * 0.8)
        X_train_lstm = X_lstm[:lstm_split]
        y_train_lstm = y_lstm[:lstm_split]
        X_test_lstm = X_lstm[lstm_split:]
        y_test_lstm = y_lstm[lstm_split:]
        
        self.train_lstm(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
        
        # Train ARIMA
        self.train_arima(train_data['USDTRY'], test_data['USDTRY'])
    
    def forecast_linear_regression(self, last_features, horizon):
        """Forecast using Linear Regression"""
        if 'LinearRegression' not in self.models:
            return None
        
        model = self.models['LinearRegression']['model']
        forecast = []
        current_features = last_features.copy()
        
        for _ in range(horizon):
            pred = model.predict([current_features])[0]
            forecast.append(pred)
            # Update features for next prediction (simplified)
            current_features = np.roll(current_features, 1)
            current_features[0] = pred
        
        return np.array(forecast)
    
    def forecast_random_forest(self, last_features, horizon):
        """Forecast using Random Forest"""
        if 'RandomForest' not in self.models:
            return None
        
        model = self.models['RandomForest']['model']
        forecast = []
        current_features = last_features.copy()
        
        for _ in range(horizon):
            pred = model.predict([current_features])[0]
            forecast.append(pred)
            # Update features for next prediction
            current_features = np.roll(current_features, 1)
            current_features[0] = pred
        
        return np.array(forecast)
    
    def forecast_lstm(self, last_sequence, horizon):
        """Forecast using LSTM"""
        if 'LSTM' not in self.models:
            return None
        
        model = self.models['LSTM']['model']
        forecast = []
        current_sequence = last_sequence.copy()
        
        for _ in range(horizon):
            # Reshape for LSTM
            input_seq = current_sequence.reshape((1, current_sequence.shape[0], 1))
            pred = model.predict(input_seq, verbose=0)[0, 0]
            forecast.append(pred)
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
        
        # Inverse transform
        forecast = np.array(forecast).reshape(-1, 1)
        forecast = self.lstm_scaler.inverse_transform(forecast).flatten()
        return forecast
    
    def forecast_arima(self, horizon):
        """Forecast using ARIMA"""
        if 'ARIMA' not in self.models or self.models['ARIMA'] is None:
            return None
        
        model = self.models['ARIMA']['model']
        forecast = model.forecast(steps=horizon)
        return forecast.values if hasattr(forecast, 'values') else forecast
    
    def generate_forecasts(self):
        """Generate forecasts for 3, 6, and 12 months"""
        print("\n" + "="*60)
        print("GENERATING FORECASTS")
        print("="*60)
        
        # Get last data point for feature-based models
        feature_data = self.create_features(self.data)
        last_row = feature_data.iloc[-1]
        feature_cols = [col for col in feature_data.columns if col != 'USDTRY']
        last_features = self.scaler.transform([last_row[feature_cols].values])[0]
        
        # Get last sequence for LSTM
        lookback = 60
        lstm_data = self.data['USDTRY'].values[-lookback:]
        scaled_lstm = self.lstm_scaler.transform(lstm_data.reshape(-1, 1)).flatten()
        
        horizons = {
            '3_months': 90,
            '6_months': 180,
            '1_year': 365
        }
        
        for horizon_name, days in horizons.items():
            print(f"\nForecasting for {horizon_name} ({days} days)...")
            forecasts = {}
            
            # Linear Regression
            lr_forecast = self.forecast_linear_regression(last_features, days)
            if lr_forecast is not None:
                forecasts['Linear Regression'] = lr_forecast
                print(f"  Linear Regression: {lr_forecast[-1]:.4f} (final forecast)")
            
            # Random Forest
            rf_forecast = self.forecast_random_forest(last_features, days)
            if rf_forecast is not None:
                forecasts['Random Forest'] = rf_forecast
                print(f"  Random Forest: {rf_forecast[-1]:.4f} (final forecast)")
            
            # LSTM
            lstm_forecast = self.forecast_lstm(scaled_lstm, days)
            if lstm_forecast is not None:
                forecasts['LSTM'] = lstm_forecast
                print(f"  LSTM: {lstm_forecast[-1]:.4f} (final forecast)")
            
            # ARIMA
            arima_forecast = self.forecast_arima(days)
            if arima_forecast is not None:
                forecasts['ARIMA'] = arima_forecast
                print(f"  ARIMA: {arima_forecast[-1]:.4f} (final forecast)")
            
            self.forecasts[horizon_name] = forecasts
    
    def visualize_results(self):
        """Visualize historical data and forecasts"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{PROJECT_NAME} - USD/TRY Exchange Rate Forecasts', fontsize=16, fontweight='bold')
        
        # Get last 365 days of historical data for context
        historical = self.data['USDTRY'].iloc[-365:]
        
        horizons = ['3_months', '6_months', '1_year']
        horizon_days = [90, 180, 365]
        titles = ['3 Months Forecast', '6 Months Forecast', '1 Year Forecast']
        
        for idx, (horizon, days, title) in enumerate(zip(horizons, horizon_days, titles)):
            ax = axes[idx]
            
            # Plot historical data
            ax.plot(historical.index, historical.values, 'b-', label='Historical', linewidth=2)
            
            # Generate future dates
            last_date = self.data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
            
            # Plot forecasts from each model
            colors = ['red', 'green', 'orange', 'purple']
            for i, (model_name, forecast) in enumerate(self.forecasts[horizon].items()):
                ax.plot(future_dates, forecast, '--', label=model_name, 
                       linewidth=2, alpha=0.8, color=colors[i % len(colors)])
            
            # Add vertical line at forecast start
            ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=2, label='Forecast Start')
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('USD/TRY Rate')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('usd_try_forecasts.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'usd_try_forecasts.png'")
        plt.show()
    
    def print_summary(self):
        """Print summary of forecasts"""
        print("\n" + "="*60)
        print("FORECAST SUMMARY")
        print("="*60)
        
        current_rate = self.data['USDTRY'].iloc[-1]
        print(f"\nCurrent USD/TRY Rate: {current_rate:.4f}")
        print(f"Date: {self.data.index[-1].date()}")
        
        for horizon_name in ['3_months', '6_months', '1_year']:
            print(f"\n{horizon_name.replace('_', ' ').title()}:")
            print("-" * 40)
            for model_name, forecast in self.forecasts[horizon_name].items():
                final_rate = forecast[-1]
                change = final_rate - current_rate
                change_pct = (change / current_rate) * 100
                print(f"  {model_name:20s}: {final_rate:8.4f} "
                      f"({change:+.4f}, {change_pct:+.2f}%)")
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE (Test Set)")
        print("="*60)
        for model_name, metrics in self.models.items():
            if metrics is not None:
                print(f"\n{model_name}:")
                if metrics['train_mae'] is not None:
                    print(f"  Train MAE: {metrics['train_mae']:.4f}")
                print(f"  Test MAE:  {metrics['test_mae']:.4f}")
                print(f"  Test R²:   {metrics['test_r2']:.4f}")


def main():
    """Main execution function"""
    print("="*60)
    print(f"{PROJECT_NAME} v{VERSION}")
    print("USD/TRY Exchange Rate Forecasting Model")
    print("="*60)
    
    # Initialize forecaster
    forecaster = USDTRYForecaster()
    
    # Fetch data
    forecaster.fetch_data(years=10)
    
    # Train all models
    forecaster.train_all_models()
    
    # Generate forecasts
    forecaster.generate_forecasts()
    
    # Print summary
    forecaster.print_summary()
    
    # Visualize results
    forecaster.visualize_results()
    
    print("\n" + "="*60)
    print("FORECASTING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

