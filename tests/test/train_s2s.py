# Python file to create Seq2Seq models with Attention based on Alpaca stock data
# Predicts Standardized Price Differences
# Author: Eli Jordan
# Date: 11 April 2025
# Gemini supported

import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for softmax
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from finta import TA
# --- Use StandardScaler ---
from sklearn.preprocessing import StandardScaler
# --------------------------
import time
import random
import os # For saving model
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import LR Scheduler

# --- Configuration ---
INPUT_SEQ_LEN = 60
OUTPUT_SEQ_LEN = 10
# --- Simplified Features ---
FEATURES = [
    "open", "high", "low", "close", "volume", "trade_count", "vwap",
    "sma15", "sma200", "rsi", "macd", "macd_signal",
    # Removed 'atr', 'roc' for now
]
# -------------------------
TARGET_FEATURE = 'close_diff' # Predict difference
ORIGINAL_PRICE_FEATURE = 'close'
CSV_PATH = "./csv/NVDA_from_2021.csv"
TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.15

# --- Model Hyperparameters ---
HIDDEN_DIM = 128
ENC_LAYERS = 2
DEC_LAYERS = 2
ENC_DROPOUT = 0.4
DEC_DROPOUT = 0.4
ATTN_DROPOUT = 0.1
LEARNING_RATE = 0.0003
NUM_EPOCHS = 200
BATCH_SIZE = 64
INITIAL_TEACHER_FORCING_RATIO = 1.0
CLIP = 1.0
WEIGHT_DECAY = 1e-4

# --- Early Stopping ---
EARLY_STOP_PATIENCE = 15
BEST_MODEL_PATH = './models/seq2seq_attn_stddiff_best_model_NVDA.pth' # << New path name

# --- Learning Rate Scheduler ---
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.1

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --------------------

# 1. Load and Preprocess Data
print("Loading and preprocessing data...")
df = pd.read_csv(CSV_PATH, index_col='timestamp', parse_dates=True)

# Add technical indicators
df["sma15"] = TA.SMA(df, 15)
df["sma200"] = TA.SMA(df, 200)
df["rsi"] = TA.RSI(df, 14, "close")
df["macd"] = TA.MACD(df)["MACD"]
df["macd_signal"] = TA.MACD(df)["SIGNAL"]
# df['atr'] = TA.ATR(df) # Removed
# df['roc'] = TA.ROC(df) # Removed

# Calculate Price Difference
df[TARGET_FEATURE] = df[ORIGINAL_PRICE_FEATURE].diff()

# Add Difference to Input Features
if TARGET_FEATURE not in FEATURES:
    FEATURES.append(TARGET_FEATURE)

# Drop rows with NaN values
df.dropna(inplace=True)

# Select features and target
data_df = df[FEATURES].copy()
target_col_index = data_df.columns.get_loc(TARGET_FEATURE)
close_col_index_in_features = FEATURES.index(ORIGINAL_PRICE_FEATURE)

# Split data
test_split_index = int(len(data_df) * (1 - TEST_SPLIT_RATIO))
train_val_df = data_df[:test_split_index]
test_df = data_df[test_split_index:]
val_split_index = int(len(train_val_df) * (1 - VALIDATION_SPLIT_RATIO))
train_df = train_val_df[:val_split_index]
val_df = train_val_df[val_split_index:]

# --- Scale using StandardScaler ---
feature_scaler = StandardScaler()
train_scaled = feature_scaler.fit_transform(train_df)
val_scaled = feature_scaler.transform(val_df)
test_scaled = feature_scaler.transform(test_df)

target_scaler = StandardScaler()
# Reshape target column for scaler's fit method
target_scaler.fit(train_df[[TARGET_FEATURE]])
# ---------------------------------

print(f"Training data shape: {train_scaled.shape}")
print(f"Validation data shape: {val_scaled.shape}")
print(f"Test data shape: {test_scaled.shape}")
print(f"Target feature '{TARGET_FEATURE}' index: {target_col_index}")
print(f"Original price feature '{ORIGINAL_PRICE_FEATURE}' index in FEATURES: {close_col_index_in_features}")


# 2. Create Dataset Class (No changes needed)
class Seq2SeqDataset(Dataset):
    def __init__(self, data, input_seq_len, output_seq_len, target_col_index):
        self.data = data
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.target_col_index = target_col_index

    def __len__(self):
        return len(self.data) - self.input_seq_len - self.output_seq_len + 1

    def __getitem__(self, idx):
        input_start = idx
        input_end = idx + self.input_seq_len
        target_start = input_end
        target_end = input_end + self.output_seq_len
        input_sequence = self.data[input_start:input_end]
        target_sequence = self.data[target_start:target_end, self.target_col_index]
        return (
            torch.tensor(input_sequence, dtype=torch.float32),
            torch.tensor(target_sequence, dtype=torch.float32).unsqueeze(-1)
        )

# 3. Define Model Architecture with Attention
class Encoder(nn.Module):
    # ... (No changes needed) ...
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)

    def forward(self, src):
        outputs, (hidden, cell) = self.lstm(src)
        return outputs, hidden, cell

class Attention(nn.Module):
    # ... (No changes needed) ...
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dropout=0.1):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy_input = torch.cat((hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(energy_input))
        attention = self.v(energy).squeeze(2)
        return self.dropout(F.softmax(attention, dim=1))


class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.lstm = nn.LSTM(enc_hid_dim + output_dim, dec_hid_dim, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # --- REMOVE Sigmoid ---
        # self.sigmoid = nn.Sigmoid()
        # ----------------------

    def forward(self, input, hidden, cell, encoder_outputs):
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        lstm_input = torch.cat((input, weighted), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        assert output.shape[1] == 1 and weighted.shape[1] == 1 and input.shape[1] == 1
        fc_input = torch.cat((output, weighted, input), dim=2)
        prediction = self.fc_out(self.dropout(fc_input))
        # --- REMOVE Sigmoid ---
        # prediction = self.sigmoid(prediction) # Output is now raw (standardized) difference
        # ----------------------
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    # ... (No changes needed) ...
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.lstm.hidden_size == decoder.dec_hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        target_len = OUTPUT_SEQ_LEN if trg is None else trg.shape[1]
        target_output_dim = self.decoder.output_dim
        outputs = torch.zeros(batch_size, target_len, target_output_dim).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        decoder_input = src[:, -1:, target_col_index].unsqueeze(-1)

        for t in range(target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t:t+1, :] = output
            teacher_force = (trg is not None) and (random.random() < teacher_forcing_ratio)
            decoder_input = trg[:, t:t+1, :] if teacher_force else output
        return outputs

# 4. Training Function (No changes needed)
def train_model(model, loader, optimizer, criterion, clip, teacher_forcing_ratio):
    # ... (No changes needed) ...
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(loader):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        loss = criterion(output, trg) # Compare predicted std diff with actual std diff
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

# 5. Evaluation Function (No changes needed)
def evaluate_model(model, loader, criterion):
    # ... (No changes needed) ...
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(loader):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            loss = criterion(output, trg) # Compare predicted std diff with actual std diff
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

# --- Modified Prediction Function for StandardScaler ---
def predict_sequence_diff(model, input_sequence_scaled, target_scaler, feature_scaler,
                          close_col_idx_in_features, target_col_idx_in_features, n_steps):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        encoder_outputs, hidden, cell = model.encoder(input_tensor)
        decoder_input = input_tensor[:, -1:, target_col_idx_in_features].unsqueeze(-1)
        predictions_scaled_diffs = []

        for _ in range(n_steps):
            output, hidden, cell = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            predictions_scaled_diffs.append(output.squeeze().item())
            decoder_input = output

    # Denormalize the predicted differences using target_scaler (StandardScaler)
    predictions_scaled_diffs = np.array(predictions_scaled_diffs).reshape(-1, 1)
    predictions_denormalized_diffs = target_scaler.inverse_transform(predictions_scaled_diffs).flatten()

    # Get the last actual absolute close price from the input sequence
    last_actual_scaled_close = input_sequence_scaled[-1, close_col_idx_in_features]

    # --- Denormalize the last actual close price using feature_scaler (StandardScaler) ---
    # Retrieve mean and scale (std dev) for the 'close' column
    close_mean = feature_scaler.mean_[close_col_idx_in_features]
    close_scale = feature_scaler.scale_[close_col_idx_in_features] # scale_ is std dev for StandardScaler
    # Formula: X = scaled * scale_ + mean_
    last_actual_absolute_close = last_actual_scaled_close * close_scale + close_mean
    # -----------------------------------------------------------------------------------

    # Reconstruct the absolute price forecast
    absolute_price_forecast = []
    current_price = last_actual_absolute_close
    for diff in predictions_denormalized_diffs:
        current_price += diff
        absolute_price_forecast.append(current_price)

    return np.array(absolute_price_forecast)
# ------------------------------------

# 7. Main Execution
if __name__ == "__main__":
    # Create Datasets and DataLoaders
    train_dataset = Seq2SeqDataset(train_scaled, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, target_col_index)
    val_dataset = Seq2SeqDataset(val_scaled, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, target_col_index)
    test_dataset = Seq2SeqDataset(test_scaled, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, target_col_index)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Instantiate Model, Loss, Optimizer
    input_dim = len(FEATURES)
    output_dim = 1

    enc_hid_dim = HIDDEN_DIM
    dec_hid_dim = HIDDEN_DIM

    attn = Attention(enc_hid_dim, dec_hid_dim, attn_dropout=ATTN_DROPOUT)
    encoder = Encoder(input_dim, enc_hid_dim, ENC_LAYERS, ENC_DROPOUT)
    decoder = Decoder(output_dim, enc_hid_dim, dec_hid_dim, DEC_LAYERS, DEC_DROPOUT, attn)

    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # --- Consider MSELoss with StandardScaler ---
    # L1Loss might be less sensitive to magnitude, which could be good or bad.
    # Let's try MSELoss again now that we're using StandardScaler
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # ------------------------------------------
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR,
                                  patience=SCHEDULER_PATIENCE, verbose=True)

    # --- Training Loop ---
    print("\nStarting Training...")
    start_time = time.time()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    teacher_forcing_ratio = INITIAL_TEACHER_FORCING_RATIO

    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        current_teacher_forcing_ratio = max(0.0, INITIAL_TEACHER_FORCING_RATIO - (epoch * (INITIAL_TEACHER_FORCING_RATIO / NUM_EPOCHS)))
        train_loss = train_model(model, train_loader, optimizer, criterion, CLIP, current_teacher_forcing_ratio)
        val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | TF Ratio: {current_teacher_forcing_ratio:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  Validation loss improved. Saved best model to {BEST_MODEL_PATH}")
        else:
            epochs_no_improve += 1
            print(f"  Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds")

    # --- Load Best Model ---
    if os.path.exists(BEST_MODEL_PATH):
        print(f"\nLoading best model from {BEST_MODEL_PATH} for prediction.")
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
    else:
        print("\nWarning: No best model found. Using model from last epoch.")

    # --- Prediction Example ---
    print("\nMaking a sample prediction using the best model...")
    sample_idx = 0
    input_seq_start_in_test = sample_idx
    input_seq_end_in_test = sample_idx + INPUT_SEQ_LEN
    actual_target_start_in_test = input_seq_end_in_test
    actual_target_end_in_test = actual_target_start_in_test + OUTPUT_SEQ_LEN

    if actual_target_end_in_test > len(test_scaled):
        print(f"Warning: Not enough data in test set for full target sequence for sample_idx={sample_idx}. Target sequence will be shorter.")
        actual_target_end_in_test = len(test_scaled)
    if input_seq_end_in_test > len(test_scaled):
         raise IndexError(f"Cannot get input sequence for sample_idx={sample_idx}.")

    sample_input_sequence_scaled = test_scaled[input_seq_start_in_test : input_seq_end_in_test]
    actual_absolute_prices = test_df[ORIGINAL_PRICE_FEATURE].iloc[actual_target_start_in_test:actual_target_end_in_test].values

    if len(actual_absolute_prices) == 0:
         raise ValueError("Failed to extract actual absolute prices for comparison.")

    predicted_absolute_prices = predict_sequence_diff(
        model,
        sample_input_sequence_scaled,
        target_scaler, # Scaler fitted on differences
        feature_scaler, # Scaler fitted on all input features
        close_col_index_in_features,
        target_col_index,
        OUTPUT_SEQ_LEN
    )

    if len(predicted_absolute_prices) > len(actual_absolute_prices):
        predicted_absolute_prices = predicted_absolute_prices[:len(actual_absolute_prices)]

    print("\n--- Sample Prediction Results (Absolute Prices) ---")
    print(f"Input sequence shape: {sample_input_sequence_scaled.shape}")
    print(f"Predicted {len(predicted_absolute_prices)} days (absolute):")
    print(np.round(predicted_absolute_prices, 2))
    print(f"\nActual {len(actual_absolute_prices)} days (absolute):")
    print(np.round(actual_absolute_prices, 2))

    if len(predicted_absolute_prices) == len(actual_absolute_prices):
        rmse = np.sqrt(np.mean((predicted_absolute_prices - actual_absolute_prices)**2))
        mae = np.mean(np.abs(predicted_absolute_prices - actual_absolute_prices))
        print(f"\nSample Prediction RMSE (Absolute): {rmse:.4f}")
        print(f"Sample Prediction MAE (Absolute): {mae:.4f}")
    else:
        print("\nCannot calculate metrics: Predicted and actual sequence lengths differ.")
