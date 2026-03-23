# Google Stock Price Prediction using LSTM

Predicts Google stock prices using a 4-layer stacked LSTM with Dropout regularization, EarlyStopping, 30-day rolling forecast, and Monte Carlo Dropout uncertainty estimation.

## Results
| Metric | Value |
|--------|-------|
| Best Val Loss | 0.000394 (Epoch 9) |
| Early Stopping | Triggered at Epoch 17 |
| Confidence | Real price stays within 95% bounds throughout |

## What I did
- Loaded 5 years of Google stock data (2012–2016) — 1258 training days, 20 test days
- Cleaned comma-formatted price strings before scaling
- Scaled Close prices using MinMaxScaler (0–1 range)
- Built sequences with 60-day lookback window → shape (1198, 60, 1)
- Trained 4-layer stacked LSTM with Dropout(0.2) after each layer
- Used EarlyStopping (patience=15, min_delta=0.0001) — stopped at epoch 17
- Evaluated with MSE, MAE, R² on test set
- Built rolling 30-day future price forecast
- Added Monte Carlo Dropout (100 iterations) for 95% confidence interval

## Model Architecture
```
Input(60, 1)
→ LSTM(50) → Dropout(0.2)
→ LSTM(50) → Dropout(0.2)
→ LSTM(50) → Dropout(0.2)
→ LSTM(50) → Dropout(0.2)
→ Dense(1)
Total params: 71,051
```

## Key Findings
- Model captures the overall upward price trend accurately
- LSTM smooths out sudden price spikes — expected behavior
- Real price stays within 95% confidence interval throughout test period
- Uncertainty bounds widen further into the future — realistic behavior
- Val loss consistently lower than train loss due to Dropout being active only during training

## How to run
1. Download `Google_Stock_Price_Train.csv` and `Google_Stock_Price_Test.csv`
2. Place both CSV files in the same folder as the notebook
3. Run all cells in order

## Output Images
| Image | Description |
|-------|-------------|
| `images/training_loss.png` | Train vs validation loss curve |
| `images/actual_vs_predicted.png` | Real vs predicted stock price |
| `images/future_forecast.png` | 30-day rolling future forecast |
| `images/uncertainty_bounds.png` | Prediction with 95% confidence interval |

## Stack
Python · TensorFlow/Keras · NumPy · Pandas · Scikit-learn · Matplotlib
```

---

## Description (GitHub About section):
```
Google stock price prediction using 4-layer stacked LSTM — EarlyStopping, 30-day rolling forecast, and Monte Carlo Dropout uncertainty estimation with 95% confidence bounds.
