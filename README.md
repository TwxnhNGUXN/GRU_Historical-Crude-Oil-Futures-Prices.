# GRU – Historical Crude Oil Futures Prices (WTI)

This repository contains a time series forecasting project that uses a **Gated Recurrent Unit (GRU)** neural network to predict the **daily closing price** of WTI crude oil futures.

## Dataset

- Source: Kaggle – *Historical Crude Oil Futures Prices (WTI and Brent)*  
- Subset used: **WTI prices**
- Main columns:
  - `date` – trading day  
  - `open`, `high`, `low`, `close` – OHLC prices for the day  
  - `volume` – trading volume (number of contracts)  
  - `average` – average price for the day  

The target variable is the **`close`** price. The other six features (`open, high, low, close, volume, average`) are used as model inputs.

## Methodology

1. **Preprocessing & EDA**
   - Convert `date` to `datetime`, sort by time and set as index.
   - Remove invalid or duplicated rows.
   - Check missing values, basic statistics and boxplots (especially for `volume` outliers).
   - Plot time series for price and volume to understand trends and volatility.

2. **Train/Validation/Test Split**
   - Split chronologically: ~70% train, 15% validation, 15% test (no shuffling).
   - Scale features and target to `[0, 1]` using `MinMaxScaler` fitted on the training set only.

3. **Sequence Construction**
   - Use a sliding window of **60 days**.
   - Each sample has shape `(60, 6)` (60 past days × 6 features) and predicts the next-day closing price.

4. **Model**
   - GRU-based neural network:
     - GRU(64, `return_sequences=True`)
     - Dropout(0.2)
     - GRU(32)
     - Dropout(0.2)
     - Dense(1) output (next-day close)
   - Loss: **MSE**, optimizer: **Adam**, metric: **MAE**.
   - **EarlyStopping** on validation loss with `patience=5` and `restore_best_weights=True`.

5. **Hyperparameter Tuning**
   - Manual grid search over:
     - window sizes `{30, 60, 90}`
     - GRU units `{32, 64}`
     - dropout `{0.2, 0.3}`
     - batch sizes `{16, 32}`
   - Best configuration selected based on lowest validation loss.

## Results

On the held-out test set, the final GRU model achieves approximately:

- **MSE:** ~14.85 (USD²)  
- **RMSE:** ~3.85 USD  
- **MAE:** ~3.09 USD  

The model captures the **overall trend** of WTI prices reasonably well, although short-term spikes and sudden shocks are harder to predict, which is expected for financial time series.

## Files

- `notebooks/` or `<your_notebook>.ipynb` – main notebook with data preprocessing, model training and evaluation.
- (Optional) `data/` – sample CSV or link/instructions to download the Kaggle dataset.

## Future Work

- Compare GRU with other models (LSTM, ARIMA, classical ML baselines).
- Include exogenous variables (macro indicators, news sentiment, etc.).
- Extend to multi-step forecasting and other energy commodities.
---
Feel free to open an issue or fork the project if you want to experiment with different architectures or datasets.
If u like my work give me star for this repo 
