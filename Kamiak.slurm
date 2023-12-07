
# Python script
python_script = """
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from DataHandler import *
from model import *
from torch.cuda.amp import GradScaler, autocast

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

# Load data
images_by_id, images_by_coordinates, path_to_coordinates = load_data()

# Flatten the data for the dataset
image_paths = []
coordinates = []
for image_path, coord in path_to_coordinates.items():
    image_paths.append(image_path)
    coordinates.append(coord)

# Initialize the dataset
dataset = ImageGPSDataset(image_paths=image_paths, coordinates=coordinates)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Function to train and validate model
def train_validate_model(model, epochs, model_name):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for images, coords in train_loader:
            images = images.to(device)
            coords = coords.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = loss_function(outputs, coords)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, coords in val_loader:
                images = images.to(device)
                coords = coords.to(device)
                with autocast():
                    outputs = model(images)
                    loss = loss_function(outputs, coords)
                total_val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {total_train_loss/len(train_loader)}, Validation Loss: {total_val_loss/len(val_loader)}')

    # Save model
    torch.save(model.state_dict(), f"{model_name}.pth")

# Train and save the ImageGPSModelV3
model_v3 = ImageGPSModelV3().to(device)
train_validate_model(model_v3, 40, "ImageGPSModelV3")

# Train and save the ResnetGPSModel
model_res = ResnetGPSModel().to(device)
train_validate_model(model_res, 20, "ResnetGPSModel")
"""

# SLURM script
slurm_script = """
#!/bin/bash
#SBATCH --job-name=gps_model_training
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=32:00:00
#SBATCH --partition=kamiak
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=32G

module load python/3.8
module load cuda/11.0
module load pytorch/1.7.0

srun python train_model.py
"""

