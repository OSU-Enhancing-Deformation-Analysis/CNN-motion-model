import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional
import random
from displacements import VectorFieldComposer
from skimage import io
import logging


def load_image(path: str | Path) -> Optional[np.ndarray]:
    """
    Load and preprocess an image for SEM analysis.

    Args:
        path: Path to the image file

    Returns:
        Normalized numpy array of the image, or None if loading fails
    """
    try:
        # Read image
        img = io.imread(path)

        # Handle different number of channels
        if img.ndim == 3:
            # Convert RGB to grayscale if needed
            img = np.mean(img, axis=2)

        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())

        return img

    except Exception as e:
        logging.error(f"Error loading image {path}: {str(e)}")
        return None


def load_image_directory(directory: str | Path) -> list[np.ndarray]:
    """
    Load all images from a directory.

    Args:
        directory: Path to directory containing images

    Returns:
        List of normalized image arrays
    """
    directory = Path(directory)
    images = []

    # Common image extensions
    extensions = ["tile_8.png"]
    # extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    for ext in extensions:
        for img_path in directory.glob(f"*{ext}"):
            img = load_image(img_path)
            if img is not None:
                images.append(img)
                print(f"Loaded {img_path.name}, shape: {img.shape}")

    print(f"Successfully loaded {len(images)} images")
    return images


class SEMSequenceDataset(Dataset):
    def __init__(self, base_images: List[np.ndarray], sequence_length: int):
        self.base_images = base_images
        self.sequence_length = sequence_length
        self.composer = VectorFieldComposer(base_images[0].shape[0])

    def generate_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        # Randomly select a base image
        base_image = random.choice(self.base_images)

        # Clear previous fields
        self.composer.clear_fields()

        # Randomly add 1-4 different fields with random parameters
        possible_fields = [
            "translation",
            "rotation",
            "shear",
            "scale",
            "gradient",
            "curl",
        ]
        n_fields = random.randint(1, 4)

        for _ in range(n_fields):
            field_type = random.choice(possible_fields)
            self.composer.add_field(
                field_type,
                amplitude=random.uniform(0.05, 0.2),
                center=(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)),
                scale=random.uniform(0.5, 1.5),
                rotation=random.uniform(0, 2 * np.pi),
            )

        # Generate sequence
        frames = []
        motion_fields = []

        for t in np.linspace(0, 1, self.sequence_length):
            # Update field parameters smoothly
            for field in self.composer.fields:
                field.amplitude *= 1 + 0.1 * np.sin(2 * np.pi * t)

            dx, dy = self.composer.compute_combined_field()
            deformed = self.composer.apply_to_image(base_image)

            frames.append(deformed)
            motion_fields.append(np.stack([dx, dy]))

        return np.array(frames), np.array(motion_fields)

    def __len__(self):
        return 100  # Number of sequences per epoch

    def __getitem__(self, idx):
        frames, motion_fields = self.generate_sequence()

        # Convert to torch tensors
        frames = torch.from_numpy(frames).float().unsqueeze(1)  # Add channel dim
        motion_fields = torch.from_numpy(motion_fields).float()

        return frames, motion_fields


class MotionPredictor(nn.Module):
    def __init__(self, sequence_length: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
        )

        self.motion_predictor = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 2, kernel_size=(1, 1, 1)),  # 2 channels for dx, dy
        )


def train_model(
    base_image_path: str,
    sequence_length: int = 20,
    epochs: int = 100,
    batch_size: int = 4,
):
    # Load base images
    base_images = load_image_directory(base_image_path)
    if not base_images:
        raise ValueError("No images were loaded successfully")

    print(f"Loaded {len(base_images)} base images")

    # Create dataset and dataloader
    dataset = SEMSequenceDataset(base_images, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    # Initialize model
    model = MotionPredictor(sequence_length).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (sequences, motion_fields) in enumerate(dataloader):
            sequences = sequences.cuda()
            motion_fields = motion_fields.cuda()

            # Forward pass
            predicted_motion = model(sequences)

            # Compute loss (L1 loss for motion vectors)
            loss = F.l1_loss(predicted_motion, motion_fields)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} complete, Average Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                f"motion_predictor_checkpoint_epoch_{epoch}.pt",
            )


if __name__ == "__main__":
    train_model(
        base_image_path="../tiles", sequence_length=20, epochs=100, batch_size=4
    )
