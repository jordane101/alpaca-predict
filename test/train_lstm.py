import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from finta import TA
import time # Added for timing comparison if needed

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --------------------

# Load the data
df = pd.read_csv("./csv/NVDA_from_2021.csv")

# Add technical indicators
df["sma15"] = TA.SMA(df, 15)
df["sma200"] = TA.SMA(df, 200)
df["rsi"] = TA.RSI(df, 14, "close")
df["macd"] = TA.MACD(df)["MACD"]
df["macd_signal"] = TA.MACD(df)["SIGNAL"]

# Drop rows with NaN values that result from calculating indicators
df.dropna(inplace=True)

# Define features and target
features = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "sma15",
    "sma200",
    "rsi",
    "macd",
    "macd_signal",
]
target = "close"

# Store the original mean and std for the target variable
target_mean = df[target].mean()
target_std = df[target].std()

# Normalize the data
for feature in features:
    df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
df[target] = (df[target] - target_mean) / target_std

# Convert to numpy arrays
data = df[features].values
labels = df[target].values

# Define sequence length
sequence_length = 60  # Increased sequence length

# Create sequences and labels
sequences = []
target_labels = []
for i in range(len(data) - sequence_length):
    sequences.append(data[i : i + sequence_length])
    target_labels.append(labels[i + sequence_length])

sequences = np.array(sequences)
target_labels = np.array(target_labels)

# Split into training and testing sets
split_ratio = 0.8  # 80% for training, 20% for testing
split_index = int(len(sequences) * split_ratio)

train_sequences = sequences[:split_index]
train_labels = target_labels[:split_index]
test_sequences = sequences[split_index:]
test_labels = target_labels[split_index:]

# Print shapes to verify
print("train_sequences shape:", train_sequences.shape)
print("train_labels shape:", train_labels.shape)
print("test_sequences shape:", test_sequences.shape)
print("test_labels shape:", test_labels.shape)

# Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels):
        # Store as numpy arrays initially, convert to tensors on the fly or pre-convert
        # Keeping them as numpy here is fine, conversion happens before model input
        self.sequences = sequences.astype(np.float32) # Ensure float32 for PyTorch
        self.labels = labels.astype(np.float32)     # Ensure float32 for PyTorch

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Return tensors directly
        return torch.from_numpy(self.sequences[idx]), torch.from_numpy(np.array(self.labels[idx])) # Ensure label is also numpy before tensor


# LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_rate)
        # Adjust linear layer input size based on bidirectional
        fc_input_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input_size, output_size)
        # Store bidirectional flag for forward pass logic if needed (already implicitly handled by layer sizes)
        self.bidirectional = bidirectional

    def forward(self, x):
        # x is expected to be on the correct device already
        # Initialize hidden and cell states on the same device as the input tensor x
        h_0 = torch.zeros(self.lstm.num_layers * (2 if self.bidirectional else 1), x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers * (2 if self.bidirectional else 1), x.size(0), self.lstm.hidden_size).to(x.device)

        # Pass through LSTM
        out, _ = self.lstm(x, (h_0, c_0))

        # Get the output from the last time step
        # out shape: (batch_size, seq_length, hidden_size * num_directions)
        out = out[:, -1, :] # Take output of the last time step

        # Apply dropout and pass through the fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        return out


# Model parameters
input_size = len(features)
hidden_size = 128
num_layers = 2
output_size = 1
dropout_rate = 0.3
bidirectional = True

# Instantiate model and move it to the designated device
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional).to(device)
criterion = nn.MSELoss()
learning_rate = 0.0005
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train() # Set model to training mode
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (sequences, labels) in enumerate(train_loader):
            # --- Move data to the selected device ---
            sequences = sequences.to(device)
            labels = labels.to(device).unsqueeze(1) # Add dimension for MSELoss and move to device
            # -----------------------------------------

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.8f}")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds")


# Prediction function
def predict_with_confidence(model, data_loader, target_mean, target_std, device):
    model.eval() # Set model to evaluation mode
    predictions = []
    actuals = []

    with torch.no_grad(): # Disable gradient calculation for inference
        for sequences, labels in data_loader:
            # --- Move input data to the selected device ---
            sequences = sequences.to(device)
            # Labels remain on CPU as they are used for comparison after prediction
            # ---------------------------------------------

            outputs = model(sequences) # Outputs are on the 'device'

            # --- Move predictions back to CPU for numpy conversion and denormalization ---
            denormalized_outputs = outputs.cpu().numpy() * target_std + target_mean
            predictions.extend(denormalized_outputs.flatten()) # Flatten in case output has extra dim

            # Denormalize the actuals (which are already on CPU)
            denormalized_labels = labels.numpy() * target_std + target_mean # labels from DataLoader are on CPU
            actuals.extend(denormalized_labels.flatten())
            # --------------------------------------------------------------------------

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse) # Often more interpretable than MSE
    print(f"Test RMSE: {rmse:.4f}")

    # Calculate confidence interval based on prediction errors
    errors = predictions - actuals
    std_error = np.std(errors)
    confidence_interval = 1.96 * std_error  # 95% confidence interval based on error distribution

    return predictions, actuals, confidence_interval


# Create Datasets and DataLoaders
batch_size = 64
train_dataset = TimeSeriesDataset(train_sequences, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False) # Optimize loading

test_dataset = TimeSeriesDataset(test_sequences, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False) # Optimize loading


# Train the model
num_epochs = 100
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# Make predictions and calculate confidence
print("\nEvaluating on test data...")
predictions, actuals, confidence_interval = predict_with_confidence(
    model, test_loader, target_mean, target_std, device
)

# Print some results (optional)
print(f"\nSample Predictions: {predictions[:10]}")
print(f"Sample Actuals: {actuals[:10]}")
print(f"95% Confidence Interval width: +/- {confidence_interval:.4f}")

# Save the model (optional)
# torch.save(model.state_dict(), 'lstm_stock_predictor.pth')
# print("Model saved to lstm_stock_predictor.pth")
