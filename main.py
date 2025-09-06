import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os

from color_prediction.model import ColorizationModel
from color_prediction.colorization_dataset import prepare_train_and_val_data
from color_prediction.utils import lab_to_rgb, save_model

def train_model():
    # Hyperparameters
    learning_rate = 1e-3
    batch_size = 64
    epochs = 50

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Prepare datasets and dataloaders
    train_dataset, val_dataset = prepare_train_and_val_data()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = ColorizationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            L_channel = data['L'].to(device)
            ab_channels = data['ab'].to(device)

            # Forward pass
            outputs = model(L_channel)
            loss = criterion(outputs, ab_channels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        print(f"Epoch {epoch+1} finished. Time taken: {end_time - start_time:.2f}s, Loss: {loss.item():.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader, desc=f"Validation {epoch+1}/{epochs}")):
                L_channel = data['L'].to(device)
                ab_channels = data['ab'].to(device)
                
                outputs = model(L_channel)
                loss = criterion(outputs, ab_channels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    print("Training complete!")
    save_model(model)

if __name__ == "__main__":
    train_model()