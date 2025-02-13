# %% [markdown]
# ### Imports

# %%
import os
import glob
import re
import random
import sys
import time
from typing import Callable, List, Tuple, Dict, TypeAlias

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from PIL import Image
import wandb

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# %% [markdown]
# ### Constants

# %%
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
GPU = torch.cuda.get_device_name(0)
print(f"Using {GPU} GPU with {GPU_MEMORY} GB of memory")

# %%

TILES_DIR = "./tiles"
# Load all images (both stem and graphite)
TILE_IMAGE_PATHS = glob.glob(os.path.join(TILES_DIR, "**/*.png"), recursive=True)
# MAX_TILES = 100  # For just quick tests
MAX_TILES = 50000  # For running all the images
NUM_TILES = min(MAX_TILES, len(TILE_IMAGE_PATHS))

TILE_SIZE = 256

# Dataset parameters
VARIATIONS_PER_IMAGE = 10

# Training parameters
# EPOCHS = 10 # Use this or MAX_TIME
# MAX_TIME = None

EPOCHS = None
# MAX_TIME = 15  # In seconds | Use this or EPOCHS
MAX_TIME = 23.5 * 60 * 60  # In seconds | Use this or EPOCHS

# ( GB - 0.5 (buffer)) / 0.13 = BATCH_SIZE
BATCH_SIZE = int((GPU_MEMORY - 1.5) / 0.13 / 2)
# BATCH_SIZE = 240  # Fills 32 GB VRAM
IMG_SIZE = TILE_SIZE
LEARNING_RATE = 0.0001
SAVE_FREQUENCY = 5  # Writes a checkpoint file

# Model name for saving files and in wandb
if len(sys.argv) < 2:
    MODEL_NAME = "b3-unknown-test"
else:
    MODEL_NAME = sys.argv[1]
MODEL_FILE = f"{MODEL_NAME}.pth"

if not os.path.exists(MODEL_NAME):
    os.makedirs(MODEL_NAME)

# %% [markdown]
# # Dataset


# %%
from dataclasses import dataclass
from scipy.ndimage import map_coordinates
from functools import wraps

farray: TypeAlias = npt.NDArray[np.float32]

# Global registry for vector fields
VECTOR_FIELDS: Dict[
    str,
    Callable[
        [farray, farray],
        Tuple[farray, farray],
    ],
] = {}


def vector_field():
    def decorator(func: Callable):
        field_name = func.__name__
        VECTOR_FIELDS[field_name] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


@vector_field()
def translation_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return np.ones_like(X), np.zeros_like(Y)


@vector_field()
def rotation_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return -Y, X


@vector_field()
def shear_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return np.ones_like(X), X


@vector_field()
def shear_field2(X, Y):
    return Y, np.zeros_like(Y)


@vector_field()
def scale_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return X, Y


@vector_field()
def gradient_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return X**2, Y**2


@vector_field()
def gradient_field2(X: farray, Y: farray) -> Tuple[farray, farray]:
    field = X**2 + Y**2
    return np.gradient(field, axis=0) * 10, np.gradient(field, axis=1) * 10


@vector_field()
def harmonic_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return np.sin(X), np.cos(Y)


@vector_field()
def harmonic_field2(X: farray, Y: farray) -> Tuple[farray, farray]:
    innerX = 2 * np.pi * X
    innerY = 2 * np.pi * Y
    return (np.sin(innerX) * np.cos(innerY)).astype(np.float32), (
        -np.cos(innerX) * np.sin(innerY)
    ).astype(np.float32)


@vector_field()
def pearling_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    r = np.sqrt(X**2 + Y**2)
    return np.sin(2 * np.pi * X) * X / (r + 1e-3), np.cos(2 * np.pi * Y) * Y / (
        r + 1e-3
    )


@vector_field()
def outward_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    r = np.sqrt(X**2 + Y**2)
    return X / (r + 1e-3), Y / (r + 1e-3)


@vector_field()
def compressing_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    r = np.sqrt(X**2 + Y**2)
    return -X / (r + 1e-3), -Y / (r + 1e-3)


@vector_field()
def vortex_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    r = np.sqrt(X**2 + Y**2)
    return -Y / (r**2 + 0.1), X / (r**2 + 0.1)


@dataclass
class VectorField:
    name: str
    field_func: Callable
    amplitude: float = 1.0
    center: Tuple[float, float] = (0, 0)
    scale: float = 1.0
    rotation: float = 0.0

    def randomize(self) -> None:
        self.center = (random.random() * 2 - 1, random.random() * 2 - 1)
        self.scale = random.random() * 2
        self.rotation = random.random() * 2 * np.pi
        # self.amplitude = random.random() * 2 - 1

    def apply(self, X: farray, Y: farray) -> Tuple[farray, farray]:
        X_centered = X - self.center[0]
        Y_centered = Y - self.center[1]

        X_scaled = X_centered * self.scale
        Y_scaled = Y_centered * self.scale

        if self.rotation != 0:
            cos_theta = np.cos(self.rotation)
            sin_theta = np.sin(self.rotation)
            X_rot = X_scaled * cos_theta - Y_scaled * sin_theta
            Y_rot = X_scaled * sin_theta + Y_scaled * cos_theta
        else:
            X_rot, Y_rot = X_scaled, Y_scaled

        dx, dy = self.field_func(X_rot, Y_rot)
        return dx * self.amplitude, dy * self.amplitude


class VectorFieldComposer:
    def __init__(self):
        self.fields: List[VectorField] = []
        # field = VectorField(name="rotation_field", field_func=rotation_field)
        # field.randomize()
        # self.fields.append(field)

    def add_field(self, field_type: str, randomize: bool = True, **kwargs) -> None:
        if field_type not in VECTOR_FIELDS:
            raise ValueError(f"Unknown field type: {field_type}")

        field = VectorField(
            name=field_type, field_func=VECTOR_FIELDS[field_type], **kwargs
        )
        if randomize:
            field.randomize()
        self.fields.append(field)

    def pop_field(self) -> None:
        self.fields.pop()

    def last(self) -> VectorField:
        return self.fields[-1]

    def compute_combined_field(self, X: farray, Y: farray) -> Tuple[farray, farray]:
        total_dx = np.zeros_like(X)
        total_dy = np.zeros_like(Y)

        for field in self.fields:
            dx, dy = field.apply(X, Y)
            total_dx += dx
            total_dy += dy

        return total_dx, total_dy

    def apply_to_image(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        width, height = image.shape

        grid_X, grid_Y = np.meshgrid(
            np.linspace(-1, 1, width), np.linspace(-1, 1, height)
        )
        dU, dV = self.compute_combined_field(grid_X, grid_Y)
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        new_x = x - dU
        new_y = y - dV

        warped_image = np.zeros_like(image)
        warped_image = map_coordinates(
            image,
            [new_y, new_x],
            order=1,
            mode="nearest",
        )

        return warped_image.astype(np.uint8)

    def display_fields(self, width: int, height: int):
        grid_X, grid_Y = np.meshgrid(
            np.linspace(-1, 1, width), np.linspace(-1, 1, height)
        )
        dU, dV = self.compute_combined_field(grid_X, grid_Y)

        dU_norm = (dU - dU.min()) / (dU.max() - dU.min())
        dV_norm = (dV - dV.min()) / (dV.max() - dV.min())

        vector_field_viz = np.zeros((*dU.shape, 3), dtype=np.uint8)
        vector_field_viz[:, :, 0] = (dU_norm * 255).astype(np.uint8)
        vector_field_viz[:, :, 1] = (dV_norm * 255).astype(np.uint8)

        return vector_field_viz


# %%
class CustomDataset(Dataset):
    def __init__(self, variations_per_image: int = 10):
        self.variations_per_image = variations_per_image

    def __len__(self):
        return NUM_TILES * self.variations_per_image

    def __getitem__(self, index):
        # Indexes work like this:
        # [1_0, ..., n_0, 1_1, ..., n_1, 1_v, ..., n_v, ...]
        # [1  , ..., n  , n+1, ..., n+n, vn+1,..., vn+n,...]
        # Where n is the number of images
        # And v is the variation number

        # Get the image index
        path_index = index % NUM_TILES

        random.seed(index)

        composer = VectorFieldComposer()

        available_fields = list(VECTOR_FIELDS.keys())
        num_fields = random.randint(1, 3)
        for _ in range(num_fields):
            field_type = random.choice(available_fields)
            composer.add_field(field_type, randomize=True)

        image = np.array(Image.open(TILE_IMAGE_PATHS[path_index], mode="r"))
        image2 = composer.apply_to_image(image)

        grid_X, grid_Y = np.meshgrid(
            np.linspace(-1, 1, TILE_SIZE), np.linspace(-1, 1, TILE_SIZE)
        )
        dx, dy = composer.compute_combined_field(grid_X, grid_Y)

        # return image.astype(np.float32), dx.astype(np.float32)
        return np.array([image, image2]).astype(np.float32), np.array([dx, dy]).astype(
            np.float32
        )


# %%
sequence_arrays = {}

# Iterate through sequence folders (e.g., "78", "g60")
for sequence_name in os.listdir(TILES_DIR):
    sequence_path = os.path.join(TILES_DIR, sequence_name)

    if os.path.isdir(sequence_path):  # Ignore hidden files/folders
        tile_arrays = {}  # Dictionary for tile arrays within this sequence

        # Iterate through image folders within the sequence (e.g., "0001.tif", "0023.tif")
        for image_folder_name in os.listdir(sequence_path):
            image_folder_path = os.path.join(sequence_path, image_folder_name)

            if os.path.isdir(image_folder_path):
                # Iterate through the tile images within the image folder
                for tile_image_name in os.listdir(image_folder_path):
                    if tile_image_name.startswith("tile_") and tile_image_name.endswith(
                        ".png"
                    ):
                        try:
                            tile_number_match = re.search(
                                r"tile_(\d+)\.png", tile_image_name
                            )
                            if tile_number_match:
                                tile_number = int(tile_number_match.group(1))
                                tile_image_path = os.path.join(
                                    image_folder_path, tile_image_name
                                )
                                if tile_number not in tile_arrays:
                                    tile_arrays[tile_number] = []
                                tile_arrays[tile_number].append(tile_image_path)

                        except ValueError:
                            print(
                                f"Warning: Could not parse tile number from {tile_image_name} in {image_folder_path}"
                            )

        sequence_arrays[sequence_name] = (
            tile_arrays  # Add the tile arrays for this sequence
        )


# %%
training_dataset = CustomDataset(VARIATIONS_PER_IMAGE)
validation_dataset = CustomDataset(VARIATIONS_PER_IMAGE)

training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

for x, y in training_dataloader:
    print(f"Shape of X [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# %% [markdown]
# # Model


# %%
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class MotionVectorRegressionNetwork(nn.Module):
    def __init__(self, input_images=2):
        super().__init__()
        # Outputs an xy motion vector per pixel
        self.input_images = input_images
        self.vector_channels = 2

        self.convolution = nn.Sequential(
            ConvolutionBlock(
                input_images, 32, kernel_size=3
            ),  # input_images (2) -> 32 channels
            nn.MaxPool2d(kernel_size=2),  # scales down by half
            ConvolutionBlock(32, 64, kernel_size=3),  # 32 -> 64 channels
            nn.MaxPool2d(kernel_size=2),  # scales down by half
            ConvolutionBlock(64, 128, kernel_size=3),  # 64 -> 128 channels
            ConvolutionBlock(128, 128, kernel_size=3),  # 128 -> 128 channels
        )

        self.output = nn.Sequential(
            # scale back up
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 128 -> 64 channels
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # 64 -> 32 channels
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                32, self.vector_channels, kernel_size=3, stride=1, padding=1
            ),  # 32 -> 2 channels
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.output(x)
        return x


model = MotionVectorRegressionNetwork(input_images=2).to(device)
print(model)


# %%
def custom_loss(predicted_vectors, target_vectors):
    l1_loss = nn.functional.l1_loss(predicted_vectors, target_vectors)
    return l1_loss


# %%
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5
)

# %%
wandb_config = {
    "gpu": GPU,
    "gpu_memory": GPU_MEMORY,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "architecture": "MotionVectorRegressionNetwork",
    "dataset": {
        "train": len(training_dataset),
        "val": len(validation_dataset),
    },
    "loss_function": "L1",
    "optimizer": "Adam",
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.1,
        "patience": 5,
        "mode": "min",
        "threshold": 0.0001,
    },
}
if MAX_TIME:
    wandb_config["max_time"] = MAX_TIME
else:
    wandb_config["epochs"] = EPOCHS

run = wandb.init(
    project="motion-model",
    name=MODEL_NAME,
    config=wandb_config,
)

run.alert(
    title=f"Start Training",
    text=f"Starting training for {MODEL_NAME}. Max time: {MAX_TIME} seconds, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, GPU Memory: {GPU_MEMORY}, GPU: {GPU}",
    level="INFO",
)

wandb.watch(model, log="all", log_freq=100)
print(run.name)

# %%
# Samples to save

samples_images = []
samples_vectors = []
seeds = [1, 3, 4, 25, 32, 38]

for s in seeds:
    images, vectors = training_dataset[s]
    samples_images.append(images)
    samples_vectors.append(vectors)


# List of tuples of (images, vectors)
samples_images = torch.from_numpy(np.array(samples_images)).float()

# %%

epoch = -1
keep_training = True

training_start_time = time.time()

while keep_training:
    epoch += 1

    print(f"Epoch {epoch+1}\n-------------------------------")
    model.train()
    epoch_training_losses = []

    size = len(training_dataloader)
    milestone = 0
    for batch, (batch_images, batch_vectors) in enumerate(training_dataloader):
        batch_images, batch_vectors = batch_images.to(device), batch_vectors.to(device)

        # Compute prediction error
        pred = model(batch_images)
        loss = custom_loss(pred, batch_vectors)

        # Backpropagation
        loss.backward()

        total_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        wandb.log(
            {
                "batch/train_loss": loss.item(),
                "batch/gradient_norm": total_grads,
                "batch/learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        epoch_training_losses.append(loss.item())

        if batch > (milestone * 100):
            milestone += 1
            print(f"loss: { loss.item():>7f}  [{batch:>5d}/{size:>5d}]")

    model.eval()
    validation_losses = []

    with torch.no_grad():
        for batch, (batch_images, batch_vectors) in enumerate(validation_dataloader):
            batch_images, batch_vectors = batch_images.to(device), batch_vectors.to(
                device
            )

            pred = model(batch_images)
            loss = custom_loss(pred, batch_vectors)
            validation_losses.append(loss.item())

    average_trianing_loss = np.mean(epoch_training_losses)
    average_validation_loss = np.mean(validation_losses)

    wandb.log(
        {
            "epoch": epoch,
            "epoch/training_loss": average_trianing_loss,
            "epoch/validation_loss": average_validation_loss,
            "epoch/learning_rate": optimizer.param_groups[0]["lr"],
        }
    )

    if epoch % SAVE_FREQUENCY == 0:
        with torch.no_grad():
            torch.save(model.state_dict(), "snapshot_save.pt")

            sample_predictions = model(samples_images.to(device))

            for i, images in enumerate(samples_images):
                vectors = samples_vectors[i]

                converted_y = vectors
                converted_y = np.vstack(
                    (
                        converted_y,
                        np.zeros((1, converted_y.shape[1], converted_y.shape[2])),
                    )
                )
                converted_y = np.transpose(converted_y, (1, 2, 0))
                converted_y = (converted_y - converted_y.min()) / (
                    converted_y.max() - converted_y.min()
                )

                converted_pred = sample_predictions[i].cpu().numpy()
                converted_pred = np.vstack(
                    (
                        converted_pred,
                        np.zeros((1, converted_pred.shape[1], converted_pred.shape[2])),
                    )
                )
                converted_pred = np.transpose(converted_pred, (1, 2, 0))
                converted_pred = (converted_pred - converted_pred.min()) / (
                    converted_pred.max() - converted_pred.min()
                )

                base_image = np.array((images[0].cpu().numpy(),) * 3)
                base_image = np.transpose(base_image, (1, 2, 0))
                morph_image = np.array((images[1].cpu().numpy(),) * 3)
                morph_image = np.transpose(morph_image, (1, 2, 0))
                combined = np.hstack(
                    (base_image, morph_image, converted_y * 256, converted_pred * 256)
                ).astype(np.uint8)

                wandb.log(
                    {
                        f"validations/sample_s{seeds[i]}": wandb.Image(
                            combined, caption=f"Epoch: {epoch}"
                        ),
                    }
                )

            sequence_name = "g69"  # g69 71-72 tile 9-11
            tiles = [9, 10, 11]
            frame_start = 71 - 35

            for tile in tiles:
                tile_sequence_paths = sequence_arrays[sequence_name][tile]
                base_image_path = tile_sequence_paths[frame_start]
                next_time_path = tile_sequence_paths[frame_start + 1]

                base_image = np.array(Image.open(base_image_path))
                next_time = np.array(Image.open(next_time_path))

                with torch.no_grad():
                    X = torch.from_numpy(np.array([base_image, next_time])).float()
                    X = X.unsqueeze(0)
                    X = X.to(device)
                    pred = model(X)

                    converted_pred = pred[0].cpu().numpy()
                    converted_pred = np.vstack(
                        (
                            converted_pred,
                            np.zeros(
                                (
                                    1,
                                    converted_pred.shape[1],
                                    converted_pred.shape[2],
                                )
                            ),
                        )
                    )
                    converted_pred = np.transpose(converted_pred, (1, 2, 0))
                    converted_pred = (converted_pred - converted_pred.min()) / (
                        converted_pred.max() - converted_pred.min()
                    )

                    base_image = np.array((X[0, 0].cpu().numpy(),) * 3)
                    base_image = np.transpose(base_image, (1, 2, 0))
                    next_time = np.array((X[0, 1].cpu().numpy(),) * 3)
                    next_time = np.transpose(next_time, (1, 2, 0))
                    combined = np.hstack(
                        (base_image, next_time, converted_pred * 256)
                    ).astype(np.uint8)

                    wandb.log(
                        {
                            f"tests/sample_{sequence_name}_{tile}_{frame_start}": wandb.Image(
                                combined, caption=f"Epoch: {epoch}"
                            ),
                        }
                    )

    scheduler.step(average_validation_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {average_trianing_loss:.4f}")
    print(f"Val Loss: {average_validation_loss:.4f}")

    if MAX_TIME:
        if time.time() - training_start_time > MAX_TIME:
            keep_training = False
            print(f"Max time reached. Stopping training.")
            break
        else:
            print(
                f"Training for {MAX_TIME - time.time() + training_start_time} more seconds."
            )
    elif EPOCHS:
        if epoch >= EPOCHS:
            keep_training = False
            print(f"Max epochs reached. Stopping training.")
            break

run.alert(
    title=f"End Training",
    text=f"Training finished for {MODEL_NAME}. Completed in {time.time() - training_start_time:.2f} seconds, Epochs: {epoch+1}",
    level="INFO",
)


model_artifact = wandb.Artifact(
    name=f"motion_vector_model_{run.id}",
    type="model",
    description="Trained motion vector model",
)

torch.save(model.state_dict(), MODEL_FILE)
model_artifact.add_file(MODEL_FILE)
wandb.log_artifact(model_artifact)

wandb.finish()

print(f"Model saved to {MODEL_FILE}")
print("Done!")
