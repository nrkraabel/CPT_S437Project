import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from DataHandler import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(config, checkpoint_dir=None, model_type="ResnetGPSModel"):
    # Data loading
    images_by_id, images_by_coordinates, path_to_coordinates = load_data()
    image_paths = []
    coordinates = []
    for image_path, coord in path_to_coordinates.items():
        image_paths.append(image_path)
        coordinates.append(coord)

    dataset = ImageGPSDataset(image_paths=image_paths, coordinates=coordinates)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]))

    # Model setup Orginally they were different now this just looks silly
    if model_type == "ResnetGPSModel":
        model = ResnetGPSModel().to(device)
    elif model_type == "ImageGPSModelV3":
        model = ImageGPSModelV3().to(device)
    else:
        raise ValueError("Invalid model type")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_function = nn.MSELoss()

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0
        for images, coords in train_loader:
            images = images.to(device)
            coords = coords.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, coords)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, coords in val_loader:
                images = images.to(device)
                coords = coords.to(device)
                outputs = model(images)
                loss = loss_function(outputs, coords)
                total_val_loss += loss.item()

        # Report metrics to Ray Tune
        tune.report(loss=total_val_loss / len(val_loader))

def runRayTune(modelType, num_samples=10, max_num_epochs=10):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64]),
        "num_epochs": tune.choice([5, 10, 20]),
        "model_type": modelType  
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(metric_columns=["loss", "training_iteration"])

    result = tune.run(
        tune.with_parameters(train_model, model_type=modelType),  
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("top trial config: {}".format(best_trial.config))
    print("Top trial final validation loss: {}".format(best_trial.last_result["loss"]))

if __name__ == "__main__":
    ray.init()
    runRayTune("ResnetGPSModel")
    runRayTune("ImageGPSModelV3")
