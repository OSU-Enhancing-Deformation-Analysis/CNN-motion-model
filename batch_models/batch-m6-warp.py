# %% [markdown]
# ### Imports

#  This is the same warping model but with the stationary AB tiles
#

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
import perlin_numpy as pnp
from collections import defaultdict
import math
from PIL import Image
from skimage.util import img_as_float
from skimage.restoration import denoise_wavelet
import wandb

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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

# SPECIFICALLY THE AB TILES
# TILES_DIR = "../../tiles/ab/"
TILES_DIR = "./tiles"
# Load all images (both stem and graphite)
TILE_IMAGE_PATHS = glob.glob(os.path.join(TILES_DIR, "**/*.png"), recursive=True)
# MAX_TILES = 100  # For just quick tests
MAX_TILES = 5000  # For running all the images
NUM_TILES = min(MAX_TILES, len(TILE_IMAGE_PATHS))

TILE_SIZE = 256

# Dataset parameters
VARIATIONS_PER_IMAGE = 1

# Training parameters
# EPOCHS = 10 # Use this or MAX_TIME
# MAX_TIME = None

EPOCHS = None
# MAX_TIME = 15  # In seconds | Use this or EPOCHS
MAX_TIME = 20 * 60 * 60  # In seconds | Use this or EPOCHS

# ( GB - 0.5 (buffer)) / 0.13 = BATCH_SIZE
BATCH_SIZE = int((GPU_MEMORY - 1.5) / 0.13 / 4)
# BATCH_SIZE = 240  # Fills 32 GB VRAM
IMG_SIZE = TILE_SIZE
LEARNING_RATE = 0.0001
SAVE_FREQUENCY = 1  # Writes a checkpoint file

# Model name for saving files and in wandb
if len(sys.argv) < 2:
    MODEL_NAME = "b4-unknown-test"
else:
    MODEL_NAME = sys.argv[1]
MODEL_FILE = f"{MODEL_NAME}.pth"

# %% [markdown]
# # Dataset
# print(TILE_IMAGE_PATHS)
GROUPED_TILE_PATHS = defaultdict(list)

for path in TILE_IMAGE_PATHS:
    # Get the folder structure above the file
    relative_path = os.path.relpath(path, TILES_DIR)
    folders = os.path.dirname(relative_path).split(os.sep)

    # Concatenate folder names to create a key
    group_key = "_".join(folders) if folders else "root"

    # Append file to the appropriate group
    GROUPED_TILE_PATHS[group_key].append(path)

# Convert defaultdict to a regular dictionary
GROUPED_TILE_PATHS = dict(GROUPED_TILE_PATHS)
GROUPED_TILE_FOLDERS = list(GROUPED_TILE_PATHS.keys())
GROUPED_TILE_INDEXES = []
tmp_x = 0

for x in GROUPED_TILE_PATHS.values():
    tmp_x += len(x)
    GROUPED_TILE_INDEXES.append(tmp_x)


# %%
from dataclasses import dataclass
from scipy.spatial import (
    ConvexHull,
)  # Correct import for ConvexHull
from scipy.ndimage import map_coordinates, gaussian_filter
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
    innerX = 0.75 * np.pi * X
    innerY = 0.75 * np.pi * Y
    sinX: farray = np.sin(innerX)
    cosX: farray = np.cos(innerX)
    sinY: farray = np.sin(innerY)
    cosY: farray = np.cos(innerY)

    return (sinX * cosY), (-cosX * sinY)


@vector_field()
def vortex_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    r = np.sqrt(X**2 + Y**2)
    return -Y / (r**2 + 0.1), X / (r**2 + 0.1)


@vector_field()
def perlin_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    noise_x = pnp.generate_perlin_noise_2d((X.shape[0], X.shape[1]), res=(1, 1))
    noise_y = pnp.generate_perlin_noise_2d((X.shape[0], X.shape[1]), res=(1, 1))
    return noise_x, noise_y


@vector_field()
def swirl_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    epsilon = 0.1  # Small smoothing factor
    radius = np.sqrt(X**2 + Y**2 + epsilon**2)  # Smoothed radius
    angle = np.arctan2(Y, X)

    magnitude = np.tanh(radius)  # Scales velocity smoothly to 0 at origin
    dx: farray = np.cos(angle + radius)
    dy: farray = np.sin(angle + radius)
    dx *= magnitude
    dy *= magnitude

    return dx, dy


@vector_field()
def vortex_field2(X: farray, Y: farray) -> Tuple[farray, farray]:
    radius = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y, X)
    dx = -Y / (radius + 0.1)
    dy = X / (radius + 0.1)
    return dx, dy


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
        self.scale = random.uniform(0.8, 2.0)
        self.rotation = random.random() * 2 * np.pi
        self.amplitude = random.uniform(0.25, 0.75)

    def apply(self, X: farray, Y: farray) -> Tuple[farray, farray]:
        X_scaled = X * self.scale
        Y_scaled = Y * self.scale

        X_centered = X_scaled - self.center[0]
        Y_centered = Y_scaled - self.center[1]

        cos_theta = np.cos(self.rotation)
        sin_theta = np.sin(self.rotation)

        X_rot_pos = X_centered * cos_theta - Y_centered * sin_theta
        Y_rot_pos = X_centered * sin_theta + Y_centered * cos_theta

        dx, dy = self.field_func(X_rot_pos, Y_rot_pos)

        X_rot = dx * cos_theta - dy * sin_theta
        Y_rot = dx * sin_theta + dy * cos_theta

        return X_rot * self.amplitude, Y_rot * self.amplitude


class VectorFieldComposer:
    def __init__(self):
        self.fields: List[VectorField] = []

        self.grid_X, self.grid_Y = np.meshgrid(
            np.linspace(-1, 1, TILE_SIZE), np.linspace(-1, 1, TILE_SIZE)
        )
        self.pos_x, self.pos_y = np.meshgrid(np.arange(TILE_SIZE), np.arange(TILE_SIZE))

    def add_field(self, field_type: str, randomize: bool = True, **kwargs) -> None:
        if field_type not in VECTOR_FIELDS:
            raise ValueError(f"Unknown field type: {field_type}")

        field = VectorField(
            name=field_type, field_func=VECTOR_FIELDS[field_type], **kwargs
        )
        if randomize:
            field.randomize()
        self.fields.append(field)

    def clear(self):
        self.fields.clear()

    def pop_field(self) -> None:
        self.fields.pop()

    def last(self) -> VectorField:
        return self.fields[-1]

    def compute_combined_field(self) -> Tuple[farray, farray]:
        total_dx = np.zeros_like(self.grid_X)
        total_dy = np.zeros_like(self.grid_Y)

        for field in self.fields:
            dx, dy = field.apply(self.grid_X, self.grid_Y)
            total_dx += dx
            total_dy += dy

        return total_dx, total_dy

    def apply_to_image(
        self, image: npt.NDArray[np.uint8]
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        dU, dV = self.compute_combined_field()

        new_x = self.pos_x - dU
        new_y = self.pos_y - dV

        warped_image = map_coordinates(
            image,
            [new_y, new_x],
            order=0,
            mode="wrap",
        )

        return warped_image.astype(np.uint8), np.array([dU, dV])


# %%


def create_square_shape(size, position=None, scale=None, rotation=None):
    """Creates a square shape with randomized position, scale, and rotation."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.75, 5.0)
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)

    square_size = int(size // 4 * scale)  # Scale the size
    square_start_x = size // 2 - square_size // 2 + position[0]  # Position the square
    square_start_y = size // 2 - square_size // 2 + position[1]

    # Clip to image boundaries
    square_start_x = max(0, min(square_start_x, size))
    square_start_y = max(0, min(square_start_y, size))
    effective_size_x = min(square_size, size - square_start_x)
    effective_size_y = min(square_size, size - square_start_y)

    if effective_size_x <= 0 or effective_size_y <= 0:
        return np.zeros((size, size), dtype=np.uint8)  # Return empty if out of bounds

    square_array = np.zeros((size, size), dtype=np.uint8)
    square_array[
        square_start_y : square_start_y + effective_size_y,
        square_start_x : square_start_x + effective_size_x,
    ] = 200

    # Rotation (simple image rotation - can be improved for shape rotation only if needed)
    if rotation != 0:
        from scipy.ndimage import rotate

        square_array = rotate(
            square_array, np.degrees(rotation), order=0, reshape=False
        )  # order=0 for nearest neighbor

    return square_array


def create_circle_shape(size, position=None, scale=None, rotation=None):
    """Creates a circle shape with randomized position and scale."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.75, 5.0)
    if rotation is None:
        rotation = 0  # Circle rotation not directly applicable

    center_x_base, center_y_base = (
        size * 3 // 4,
        size // 4,
    )  # Base center, will be offset
    center_x = int(center_x_base + position[0])
    center_y = int(center_y_base + position[1])
    radius = int(size // 8 * scale)  # Scale radius

    circle_array = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2:
                circle_array[y, x] = 150
    return circle_array


def create_blob_shape(size, position=None, scale=None, rotation=None):
    """Creates a blob shape with randomized position and scale."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.5, 1.5)
    if rotation is None:
        rotation = 0  # Blob rotation not directly applicable

    center_x_base, center_y_base = size // 2, size // 2  # Base center, will be offset
    center_x = int(center_x_base + position[0])
    center_y = int(center_y_base + position[1])
    scaled_size = int(size * scale)  # Scale the effective size for blob radius

    blob_array = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            value = int(
                255 * (1 - distance_from_center / (scaled_size / 2))
            )  # Scaled size for radius
            value = max(0, min(255, value))
            blob_array[y, x] = value
    return blob_array


def create_swirl_shape(size, position=None, scale=None, rotation=None):
    """Creates a swirl pattern with randomized position, scale, and rotation."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(4.0, 7.0)
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)

    swirl_array = np.zeros((size, size), dtype=np.uint8)
    center_x_base, center_y_base = size // 2, size // 2
    center_x = center_x_base + position[0]
    center_y = center_y_base + position[1]

    for y in range(size):
        for x in range(size):
            x_rel = (x - center_x) / scale
            y_rel = (y - center_y) / scale

            if rotation != 0:
                cos_theta = np.cos(rotation)
                sin_theta = np.sin(rotation)
                x_rot = x_rel * cos_theta - y_rel * sin_theta
                y_rot = x_rel * sin_theta + y_rel * cos_theta
                x_rel, y_rel = x_rot, y_rot

            angle = np.arctan2(y_rel, x_rel)
            radius = np.sqrt(x_rel**2 + y_rel**2 + 0.1)

            value = int(128 + 127 * np.sin((radius / 5) + angle * 5))
            swirl_array[y, x] = value

    return swirl_array


def create_gradient_shape(size, position=None, scale=None, rotation=None):
    """Creates a gradient shape with randomized position, scale, and rotation (mostly scale/position effect)."""
    if position is None:
        position_x = random.randint(
            -size // 2, size // 2
        )  # Wider position range for gradient
        position_y = random.randint(-size // 2, size // 2)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.5, 2.0)  # Wider scale range for gradient
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)  # Rotation for gradient direction

    gradient_array = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            x_rel = (
                x + position[0]
            ) / scale  # Position and scale affect gradient position and spread
            y_rel = (y + position[1]) / scale

            if rotation != 0:  # Rotate gradient direction
                cos_theta = np.cos(rotation)
                sin_theta = np.sin(rotation)
                x_rot = x_rel * cos_theta - y_rel * sin_theta
                x_rel = x_rot  # Only horizontal gradient, so only rotate x coordinate

            value = int(
                255 * ((x_rel + size) % size) / size
            )  # Modulo for repeating gradient if scaled up
            gradient_array[y, x] = value
    return gradient_array


def create_checkers_shape(
    size, checker_size_param=None, position=None, scale=None, rotation=None
):
    """Creates a checkerboard pattern with randomized checker size, position, scale, and rotation."""
    if checker_size_param is None:
        checker_size_param = random.randint(4, 16)  # Random checker size
    if position is None:
        position_x = random.randint(
            -size // 2, size // 2
        )  # Wider position range for checkers
        position_y = random.randint(-size // 2, size // 2)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(2.0, 6.0)  # Wider scale range for checkers
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)  # Rotation for checker orientation

    checkers_array = np.zeros((size, size), dtype=np.uint8)
    scaled_checker_size = int(checker_size_param * scale)  # Scale checker size

    for y in range(size):
        for x in range(size):
            x_rel = (x + position[0]) / scale  # Position and scale affect checker grid
            y_rel = (y + position[1]) / scale

            if rotation != 0:  # Rotate checker grid
                cos_theta = np.cos(rotation)
                sin_theta = np.sin(rotation)
                x_rot = x_rel * cos_theta - y_rel * sin_theta
                y_rot = x_rel * sin_theta + y_rel * cos_theta
                x_rel, y_rel = x_rot, y_rot

            checker_x = int(x_rel) // scaled_checker_size
            checker_y = int(y_rel) // scaled_checker_size

            if ((checker_x) + (checker_y)) % 2 == 0:
                checkers_array[y, x] = 200  # Light gray
            else:
                checkers_array[y, x] = 50  # Dark gray
    return checkers_array


def create_rectangle_shape(size, position=None, scale=None, rotation=None):
    """Creates a rectangle shape with randomized position, scale, and rotation."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.75, 5.0)
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)

    rect_width = int(size // 2 * scale)  # Scale width
    rect_height = int(size // 3 * scale)  # Scale height
    start_x = size // 2 - rect_width // 2 + position[0]  # Position rectangle
    start_y = size // 2 - rect_height // 2 + position[1]

    # Clip to image boundaries
    start_x = max(0, min(start_x, size))
    start_y = max(0, min(start_y, size))
    effective_width = min(rect_width, size - start_x)
    effective_height = min(rect_height, size - start_y)

    if effective_width <= 0 or effective_height <= 0:
        return np.zeros((size, size), dtype=np.uint8)  # Return empty if out of bounds

    rect_array = np.zeros((size, size), dtype=np.uint8)
    rect_array[
        start_y : start_y + effective_height, start_x : start_x + effective_width
    ] = 220  # Slightly lighter gray

    # Rotation (simple image rotation)
    if rotation != 0:
        from scipy.ndimage import rotate

        rect_array = rotate(rect_array, np.degrees(rotation), order=0, reshape=False)

    return rect_array


def create_triangle_shape(size, position=None, scale=None, rotation=None):
    """Creates a triangle shape with randomized position, scale, and rotation."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.5, 1.5)
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)

    triangle_array = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            x_rel = (
                x - size // 2 - position[0]
            ) / scale  # Scale and position relative to center
            y_rel = (y - size // 2 - position[1]) / scale

            if rotation != 0:  # Rotate coordinates
                cos_theta = np.cos(rotation)
                sin_theta = np.sin(rotation)
                x_rot = x_rel * cos_theta - y_rel * sin_theta
                y_rot = x_rel * sin_theta + y_rel * cos_theta
                x_rel, y_rel = x_rot, y_rot

            if (
                y_rel >= x_rel
            ):  # Original condition relative to *transformed* coordinates
                triangle_array[y, x] = 180
    return triangle_array


def create_ellipse_shape(size, position=None, scale=None, rotation=None):
    """Creates an ellipse shape with randomized position, scale, and rotation."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.75, 5.0)
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)

    center_x_base, center_y_base = size // 2, size // 2  # Base center for ellipse
    center_x = int(center_x_base + position[0])
    center_y = int(center_y_base + position[1])
    major_axis = size // 3 * scale  # Scale axes
    minor_axis = size // 4 * scale

    ellipse_array = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            x_rel = x - center_x
            y_rel = y - center_y

            if rotation != 0:  # Rotate coordinates
                cos_theta = np.cos(rotation)
                sin_theta = np.sin(rotation)
                x_rot = x_rel * cos_theta - y_rel * sin_theta
                y_rot = x_rel * sin_theta + y_rel * cos_theta
                x_rel, y_rel = x_rot, y_rel

            if ((x_rel) ** 2 / major_axis**2) + ((y_rel) ** 2 / minor_axis**2) <= 1:
                ellipse_array[y, x] = 120
    return ellipse_array


def create_pentagon_shape(size, position=None, scale=None, rotation=None):
    """Creates a pentagon shape with randomized position, scale, and rotation."""
    return create_polygon_shape(size, 5, position, scale, rotation)


def create_hexagon_shape(size, position=None, scale=None, rotation=None):
    """Creates a hexagon shape with randomized position, scale, and rotation."""
    return create_polygon_shape(size, 6, position, scale, rotation)


def create_octagon_shape(size, position=None, scale=None, rotation=None):
    """Creates an octagon shape with randomized position, scale, and rotation."""
    return create_polygon_shape(size, 8, position, scale, rotation)


def create_polygon_shape(size, num_sides, position=None, scale=None, rotation=None):
    """Creates a regular polygon shape with randomized position, scale, and rotation."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.75, 5.0)
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)

    polygon_array = np.zeros((size, size), dtype=np.uint8)
    center_x_base, center_y_base = size // 2, size // 2  # Base center for polygon
    center_x = center_x_base + position[0]
    center_y = center_y_base + position[1]
    radius = size * 0.4 * scale  # Scale radius

    points = []
    for i in range(num_sides):
        angle = 2 * math.pi * i / num_sides
        x_base = center_x + radius * math.cos(angle)
        y_base = center_y + radius * math.sin(angle)

        x_rel = x_base - center_x  # Coordinates relative to center for rotation
        y_rel = y_base - center_y

        if rotation != 0:  # Rotate vertices
            cos_theta = np.cos(rotation)
            sin_theta = np.sin(rotation)
            x_rot = x_rel * cos_theta - y_rel * sin_theta
            y_rot = x_rel * sin_theta + y_rel * cos_theta
            x_rel, y_rel = x_rot, y_rel

        x = int(center_x + x_rel)  # Add back center offset
        y = int(center_y + y_rel)
        points.append((x, y))

    return create_polygon_shape_vertices(size, points, color=230)


def create_random_convex_shape(size, position=None, scale=None, rotation=None):
    """Creates a random convex shape with randomized position, scale, and rotation."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.75, 5.0)
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)

    num_points = random.randint(5, 10)
    points_base = np.random.randint(
        0, size, size=(num_points, 2)
    )  # Base points before transform

    points_transformed = []
    center_x, center_y = (
        size // 2 + position[0],
        size // 2 + position[1],
    )  # Adjusted center

    for point in points_base:
        x_rel = (point[0] - size // 2) * scale  # Scale relative to center
        y_rel = (point[1] - size // 2) * scale

        if rotation != 0:  # Rotate points
            cos_theta = np.cos(rotation)
            sin_theta = np.sin(rotation)
            x_rot = x_rel * cos_theta - y_rel * sin_theta
            y_rot = x_rel * sin_theta + y_rel * cos_theta
            x_rel, y_rel = x_rot, y_rot

        x = int(center_x + x_rel)  # Position after transform
        y = int(center_y + y_rel)
        points_transformed.append((x, y))

    try:
        hull = ConvexHull(points_transformed)
        vertices = []
        for vertex_index in hull.vertices:
            vertices.append(
                (
                    int(points_transformed[vertex_index][0]),
                    int(points_transformed[vertex_index][1]),
                )
            )
        return create_polygon_shape_vertices(size, vertices, color=160)
    except:  # ConvexHull can fail for collinear points, handle gracefully
        return create_square_shape(size)  # Fallback to a simple shape


def create_random_concave_shape(size, position=None, scale=None, rotation=None):
    """Creates a random concave shape with randomized position, scale, and rotation."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.75, 5.0)
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)

    convex_shape = create_random_convex_shape(
        size, position, scale, rotation
    )  # Pass parameters
    vertices_convex = []
    num_points = random.randint(5, 10)
    points_convex = np.random.randint(0, size, size=(num_points, 2))

    points_transformed = []
    center_x, center_y = (
        size // 2 + position[0],
        size // 2 + position[1],
    )  # Adjusted center

    for point in points_convex:
        x_rel = (point[0] - size // 2) * scale  # Scale relative to center
        y_rel = (point[1] - size // 2) * scale

        if rotation != 0:  # Rotate points
            cos_theta = np.cos(rotation)
            sin_theta = np.sin(rotation)
            x_rot = x_rel * cos_theta - y_rel * sin_theta
            y_rot = x_rel * sin_theta + y_rel * cos_theta
            x_rel, y_rel = x_rot, y_rel

        x = int(center_x + x_rel)  # Position after transform
        y = int(center_y + y_rel)
        points_transformed.append((x, y))

    try:
        hull_convex = ConvexHull(points_transformed)
        for vertex_index in hull_convex.vertices:
            vertices_convex.append(
                np.array(
                    [
                        points_transformed[vertex_index][0],
                        points_transformed[vertex_index][1],
                    ]
                )
            )
    except:
        return create_square_shape(size)  # Fallback if convex hull fails

    vertices_concave = list(vertices_convex)

    num_dents = random.randint(1, min(3, len(vertices_convex)))
    dent_indices = random.sample(range(len(vertices_convex)), num_dents)

    center_x_dent, center_y_dent = (
        size // 2,
        size // 2,
    )  # Center for dent direction calculation - not transformed
    for index in dent_indices:
        vertex = vertices_concave[index]
        direction_to_center = (
            np.array([center_x_dent, center_y_dent]) - vertex
        )  # Dent direction towards original center
        direction_to_center = (
            direction_to_center / np.linalg.norm(direction_to_center)
            if np.linalg.norm(direction_to_center) > 0
            else np.array([0, 0])
        )
        dent_amount = random.uniform(0, size / 8)
        vertices_concave[index] = vertex + direction_to_center * dent_amount

    vertices_concave_tuples = [(int(v[0]), int(v[1])) for v in vertices_concave]
    return create_polygon_shape_vertices(size, vertices_concave_tuples, color=240)


def create_polygon_shape_vertices(size, vertices, color):
    polygon_array = np.zeros((size, size), dtype=np.uint8)
    min_y = min(p[1] for p in vertices)
    max_y = max(p[1] for p in vertices)

    # Clipping vertices to image boundaries (important for rotated/scaled shapes)
    clipped_vertices = []
    for x, y in vertices:
        clipped_x = max(0, min(x, size - 1))
        clipped_y = max(0, min(y, size - 1))
        clipped_vertices.append((clipped_x, clipped_y))

    min_y = min(p[1] for p in clipped_vertices)
    max_y = max(p[1] for p in clipped_vertices)
    if min_y >= size or max_y < 0:  # Polygon is completely outside, return empty
        return polygon_array

    for y in range(
        max(0, min_y), min(size, max_y + 1)
    ):  # Iterate only within valid y range
        x_coords = []
        num_vertices = len(clipped_vertices)
        for i in range(num_vertices):
            p1 = clipped_vertices[i]
            p2 = clipped_vertices[(i + 1) % num_vertices]
            if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                if p1[1] != p2[1]:
                    x_intersection = int(
                        p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                    )
                    x_coords.append(x_intersection)
        x_coords.sort()
        for i in range(0, len(x_coords), 2):
            x_start = max(0, min(x_coords[i], size)) if i < len(x_coords) else 0
            x_end = max(0, min(x_coords[i + 1], size)) if i + 1 < len(x_coords) else 0
            if x_start < x_end:
                polygon_array[y, x_start:x_end] = color
    return polygon_array


def create_radial_pattern_shape(size, position=None, scale=None, rotation=None):
    """Creates a radial pattern with randomized position, scale, and rotation."""
    if position is None:
        position_x = random.randint(-size // 2, size // 2)
        position_y = random.randint(-size // 2, size // 2)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(12.0, 20.0)
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)

    radial_array = np.zeros((size, size), dtype=np.uint8)
    center_x_base, center_y_base = (
        size // 2,
        size // 2,
    )  # Base center for radial pattern
    center_x = center_x_base + position[0]
    center_y = center_y_base + position[1]

    for y in range(size):
        for x in range(size):
            x_rel = (x - center_x) / scale  # Scale coordinates relative to center
            y_rel = (y - center_y) / scale

            if rotation != 0:  # Rotate coordinates
                cos_theta = np.cos(rotation)
                sin_theta = np.sin(rotation)
                x_rot = x_rel * cos_theta - y_rel * sin_theta
                y_rot = x_rel * sin_theta + y_rel * cos_theta
                x_rel, y_rel = x_rot, y_rot

            distance_from_center = np.sqrt(x_rel**2 + y_rel**2)
            value = int(
                128 + 127 * math.sin(distance_from_center / 5 * 2 * math.pi)
            )  # Adjust frequency for bands
            radial_array[y, x] = value
    return radial_array


def create_wave_pattern_shape(size, position=None, scale=None, rotation=None):
    """Creates a wave pattern with randomized position, scale, and rotation."""
    if position is None:
        position_x = random.randint(-size // 2, size // 2)
        position_y = random.randint(-size // 2, size // 2)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(10.0, 15.0)
    if rotation is None:
        rotation = random.uniform(0, 2 * np.pi)

    wave_array = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            x_rel = (x + position[0]) / scale  # Position and scale wave pattern

            if (
                rotation != 0
            ):  # Rotate wave direction (affects horizontal wave in this case)
                cos_theta = np.cos(rotation)
                sin_theta = np.sin(rotation)
                x_rot = (
                    x_rel * cos_theta - (y + position[1]) / scale * sin_theta
                )  # Approximate rotation for horizontal wave
                x_rel = x_rot  # Apply rotation mainly to x

            value = int(
                128 + 127 * math.sin(x_rel / 8 * 2 * math.pi)
            )  # Adjust frequency for waves
            wave_array[y, x] = value
    return wave_array


def create_perlin_noise_shape(
    size,
    octaves=None,
    persistence=None,
    lacunarity=None,
    seed=None,
    position=None,
    scale=None,
    rotation=None,
    res=8,
):  # Corrected lacunarity type
    """Creates a Perlin noise pattern with randomized parameters, position, scale, and rotation (scale/position effect)."""
    if position is None:
        position_x = random.randint(-size // 4, size // 4)
        position_y = random.randint(-size // 4, size // 4)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.5, 2.0)
    if rotation is None:
        rotation = random.uniform(
            0, 2 * np.pi
        )  # Rotation for noise orientation (image rotation)

    res = random.choice([2, 4, 8])

    scaled_size_float = size * scale  # Calculate scaled size as float first
    scaled_size = int(scaled_size_float)  # Convert to int
    scaled_size = (
        scaled_size // res
    ) * res  # Ensure scaled_size is a multiple of res (integer division then multiply)
    if scaled_size % 2 == 1:
        scaled_size -= 1
    if scaled_size == 0:  # Handle case where scaling makes size too small
        scaled_size = res  # Ensure it's at least res if scale is very small

    offset_x = position[0] + (size - scaled_size) // 2  # Position offset
    offset_y = position[1] + (size - scaled_size) // 2

    if seed is not None:
        np.random.seed(seed)  # Seed for reproducibility

    try:
        noise = pnp.generate_perlin_noise_2d((scaled_size, scaled_size), res=(res, res))
    except Exception as e:
        print(f"Error generating Perlin noise: {e}")
        return create_blob_shape(size)  # Return fallback checkerboard

    normalized_noise_scaled = ((noise + 1) / 2 * 255).astype(
        np.uint8
    )  # Noise generated on scaled size

    perlin_array = np.zeros((size, size), dtype=np.uint8)  # Final image is full size
    # Paste scaled noise into final image with offset
    start_y = max(0, offset_y)
    start_x = max(0, offset_x)
    end_y = min(size, offset_y + scaled_size)
    end_x = min(size, offset_x + scaled_size)

    source_start_y = 0 if offset_y >= 0 else -offset_y
    source_start_x = 0 if offset_x >= 0 else -offset_x

    target_height = end_y - start_y
    target_width = end_x - start_x

    source_end_y = source_start_y + target_height
    source_end_x = source_start_x + target_width

    perlin_array[start_y:end_y, start_x:end_x] = normalized_noise_scaled[
        source_start_y:source_end_y, source_start_x:source_end_x
    ]

    if rotation != 0:  # Rotate the entire image
        from scipy.ndimage import rotate

        perlin_array = rotate(
            perlin_array, np.degrees(rotation), order=0, reshape=False
        )

    return perlin_array


def create_stripes_pattern_shape(
    size, stripe_width_param=None, angle=None, position=None, scale=None
):
    """Creates a stripes pattern with randomized stripe width, angle, position, and scale (scale/position effect)."""
    if stripe_width_param is None:
        stripe_width_param = random.randint(4, 16)  # Random stripe width
    if angle is None:
        angle = random.uniform(0, 90)  # Random angle (vertical or horizontal)
    if position is None:
        position_x = random.randint(-size // 2, size // 2)
        position_y = random.randint(-size // 2, size // 2)
        position = (position_x, position_y)
    if scale is None:
        scale = random.uniform(0.75, 5.0)

    stripes_array = np.zeros((size, size), dtype=np.uint8)
    scaled_stripe_width = int(stripe_width_param * scale)  # Scale stripe width

    for y in range(size):
        for x in range(size):
            x_rel = (x + position[0]) / scale  # Position and scale stripe pattern
            y_rel = (y + position[1]) / scale

            if angle == 0:  # Vertical stripes
                stripe_index = int(x_rel) // scaled_stripe_width
            else:  # Horizontal stripes
                stripe_index = int(y_rel) // scaled_stripe_width

            if stripe_index % 2 == 0:
                stripes_array[y, x] = 200  # Light gray stripe
            else:
                stripes_array[y, x] = 50  # Dark gray stripe
    return stripes_array


# --- Updated Shape Function List with Variations ---
shape_functions = [
    create_square_shape,
    create_circle_shape,
    create_blob_shape,
    create_swirl_shape,
    create_gradient_shape,
    create_checkers_shape,
    create_rectangle_shape,
    create_triangle_shape,
    create_ellipse_shape,
    create_pentagon_shape,
    create_hexagon_shape,
    create_octagon_shape,
    create_random_convex_shape,
    create_random_concave_shape,
    create_radial_pattern_shape,
    create_wave_pattern_shape,
    create_perlin_noise_shape,
    create_stripes_pattern_shape,
]


def extract_wavelet_noise(image):
    denoised_image = denoise_wavelet(image, rescale_sigma=True)
    noise_image = image - denoised_image
    return denoised_image, noise_image


# %%
current_epoch = 0


class CustomDataset(Dataset):
    def __init__(self, variations_per_image: int = 10, validate: bool = False):
        self.variations_per_image = variations_per_image
        self.validate = validate

        self.composer = VectorFieldComposer()
        self.available_fields = list(VECTOR_FIELDS.keys())

        self.pos_x, self.pos_y = np.meshgrid(np.arange(TILE_SIZE), np.arange(TILE_SIZE))

    def __len__(self):
        return NUM_TILES * self.variations_per_image

    def __getitem__(self, index):
        # Indexes work like this:
        # [1_0, ..., n_0, 1_1, ..., n_1, 1_v, ..., n_v, ...]
        # [1  , ..., n  , n+1, ..., n+n, vn+1,..., vn+n,...]
        # Where n is the number of images
        # And v is the variation number
        """
        Indexes work like this:
        Data: [1, 2, 3, 4] [1, 2, 3, 4] [1, 2, 3, 4] [1, 2, 3, 4]
        Index: 0---  2---   3---  5---   6---  8---   9---  11--
                  1---         4---         7---         10--
        """

        global current_epoch

        image_index = index % NUM_TILES
        image_chunk = None
        image_index_in_chunk = 0
        for i, x in enumerate(GROUPED_TILE_INDEXES):
            if image_index < x:
                image_chunk = GROUPED_TILE_FOLDERS[i]
                if i == 0:
                    image_index_in_chunk = image_index
                else:
                    image_index_in_chunk = image_index - GROUPED_TILE_INDEXES[i - 1]
                break

        if image_chunk is None:
            image_chunk = GROUPED_TILE_FOLDERS[0]
            image_index_in_chunk = 0

        image1_path = GROUPED_TILE_PATHS[image_chunk][image_index_in_chunk]
        image2_path = GROUPED_TILE_PATHS[image_chunk][image_index_in_chunk - 1]

        # Get the image index
        if self.validate:
            index += 10000000
            random.seed(index)
        else:
            random.seed(index + (current_epoch * NUM_TILES * self.variations_per_image))

        self.composer.clear()

        num_fields = random.randint(1, 2)
        for _ in range(num_fields):
            field_type = random.choice(self.available_fields)
            self.composer.add_field(field_type, randomize=True)

        computed_field = np.array(self.composer.compute_combined_field())

        # New Variations
        if random.random() > 0.4:
            # Adding a shape layer
            shape_function = random.choice(shape_functions)
            shape_layer = shape_function(TILE_SIZE)  # Image as uint8 0-255

            if random.random() > 0.5:
                shape_layer = 255 - shape_layer  # Invert mask

            # Morph the image
            self.composer.clear()
            shape_morph_composer = self.composer
            num_fields = random.randint(1, 2)
            for _ in range(num_fields):
                field_type = random.choice(self.available_fields)
                shape_morph_composer.add_field(field_type, randomize=True)

            morphed_shape, _field = shape_morph_composer.apply_to_image(shape_layer)
            if random.random() > 0.3:
                morphed_shape = gaussian_filter(morphed_shape, sigma=1)
                morphed_shape = (morphed_shape * (255 / np.max(morphed_shape))).astype(
                    np.uint8
                )  # Normalize after blurring

            morphed_shape = morphed_shape.astype(np.float32) / 255

            if random.random() > 0.5:
                # Invert the inner region
                final_field = (1 - morphed_shape) * (
                    computed_field * -1
                ) + morphed_shape * computed_field
            else:
                # Put another field in the inner region

                another_vector_field = VectorFieldComposer()
                num_fields = random.randint(1, 2)
                for _ in range(num_fields):
                    field_type = random.choice(self.available_fields)
                    another_vector_field.add_field(field_type, randomize=True)

                another_computed_field = np.array(
                    another_vector_field.compute_combined_field()
                )

                final_field = (
                    1 - morphed_shape
                ) * computed_field + morphed_shape * another_computed_field
        else:
            final_field = computed_field

        image1 = np.array(Image.open(TILE_IMAGE_PATHS[image1_path], mode="r"))
        image2 = np.array(Image.open(TILE_IMAGE_PATHS[image2_path], mode="r"))

        dU, dV = final_field

        new_x = self.pos_x - dU
        new_y = self.pos_y - dV

        denoised_image, extracted_noise = extract_wavelet_noise(img_as_float(image2))
        warped_image = map_coordinates(
            denoised_image,
            [new_y, new_x],
            order=0,
            mode="wrap",
        )
        applied_denoised_image = np.clip(warped_image + extracted_noise, 0, 1)

        return np.array([image1, applied_denoised_image]).astype(np.float32), np.array(
            [dU, dV]
        ).astype(np.float32)


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
NUM_WORKERS = 4
training_dataset = CustomDataset(VARIATIONS_PER_IMAGE)
validation_dataset = CustomDataset(VARIATIONS_PER_IMAGE, True)

training_dataloader = DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

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


class CorrelationLayer(nn.Module):
    def __init__(
        self, patch_size=1, kernel_size=1, stride=1, padding=0, max_displacement=20
    ):
        super().__init__()
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_displacement = max_displacement

    def forward(self, feature_map_1, feature_map_2):
        """
        Args:
            feature_map_1: Feature map from image 1 (N, C, H, W)
            feature_map_2: Feature map from image 2 (N, C, H, W)

        Returns:
            correlation_volume: Correlation volume (N, 1, H, W, (2*max_displacement+1)**2)
        """
        batch_size, channels, height, width = feature_map_1.size()

        # Pad feature_map_2 to handle displacements
        padding_size = self.max_displacement
        feature_map_2_padded = F.pad(
            feature_map_2, (padding_size, padding_size, padding_size, padding_size)
        )

        correlation_volume_list = []

        # loop over all possible displacements within max_displacement
        for displacement_y in range(-self.max_displacement, self.max_displacement + 1):
            for displacement_x in range(
                -self.max_displacement, self.max_displacement + 1
            ):
                # Shift feature_map_2
                shifted_feature_map_2 = feature_map_2_padded[
                    :,
                    :,
                    padding_size
                    + displacement_y : padding_size
                    + displacement_y
                    + height,
                    padding_size
                    + displacement_x : padding_size
                    + displacement_x
                    + width,
                ]

                # now we compute correlation and reshape for correlation volume
                correlation_map = (feature_map_1 * shifted_feature_map_2).sum(
                    dim=1, keepdim=True
                )  # Sum over channels
                correlation_volume_list.append(correlation_map)

        # put them all together
        correlation_volume = torch.cat(
            correlation_volume_list, dim=1
        )  # N, (2*max_displacement+1)**2, H, W
        correlation_volume = correlation_volume.permute(0, 2, 3, 1).unsqueeze(
            1
        )  # N, 1, H, W, (2*max_displacement+1)**2 - reshape to match expected output

        return correlation_volume


class MotionVectorRegressionNetworkWithCorrelation(nn.Module):
    def __init__(self, input_images=2, max_displacement=20):
        super().__init__()
        # Outputs an xy motion vector per pixel
        self.input_images = input_images
        self.vector_channels = 2
        self.max_displacement = max_displacement  # Store max_displacement

        self.feature_convolution = nn.Sequential(
            ConvolutionBlock(1, 32, kernel_size=3),  # input_images (1) -> 32 channels
            nn.MaxPool2d(kernel_size=2),  # scales down by half
            ConvolutionBlock(32, 64, kernel_size=3),  # 32 -> 64 channels
            nn.MaxPool2d(kernel_size=2),  # scales down by half
            ConvolutionBlock(64, 128, kernel_size=3),  # 64 -> 128 channels
            # ConvolutionBlock(64, 128, kernel_size=3),
        )

        self.correlation_layer = CorrelationLayer(
            max_displacement=self.max_displacement
        )  # Correlation Layer

        self.convolution_after_correlation = (
            nn.Sequential(  # Convolution layers after correlation
                ConvolutionBlock(
                    128 + (2 * max_displacement + 1) ** 2, 128, kernel_size=3
                ),
                ConvolutionBlock(128, 128, kernel_size=3),  # 128 -> 128 channels
            )
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
        # split input into image 1 and image 2
        image1 = x[:, 0:1, :, :]
        image2 = x[:, 1:2, :, :]

        features1 = self.feature_convolution(image1)
        features2 = self.feature_convolution(image2)

        # Correlation layer
        correlation_volume = self.correlation_layer(
            features1, features2
        )  # N, 1, H, W, (2*max_displacement+1)**2

        # concatenate correlation volume with features1 (you can experiment with features2 or concatenation strategy)
        # reshape correlation volume to (N, C, H, W) where C = (2*max_displacement+1)**2
        correlation_volume_reshaped = correlation_volume.squeeze(1).permute(
            0, 3, 1, 2
        )  # N, (2*max_displacement+1)**2, H, W
        combined_features = torch.cat(
            (features1, correlation_volume_reshaped), dim=1
        )  # Concatenate along channel dimension

        x = self.convolution_after_correlation(combined_features)
        x = self.output(x)  # output layers
        return x


class MotionVectorRegressionNetworkWithWarping(
    nn.Module
):  # Model 3: Model 1 + Warping (2-Stage Stacked)
    def __init__(self, input_images=2, max_displacement=20):
        super().__init__()
        self.input_images = input_images
        self.max_displacement = max_displacement

        self.stage1_model = MotionVectorRegressionNetworkWithCorrelation(
            input_images=input_images, max_displacement=self.max_displacement
        )
        self.stage2_model = MotionVectorRegressionNetworkWithCorrelation(
            input_images=input_images, max_displacement=self.max_displacement
        )  # Stage 2 model - same architecture as stage 1

    def forward(self, x):
        # Split input into image 1 and image 2
        image1 = x[:, 0:1, :, :]  # Assuming grayscale input, adjust if RGB
        image2 = x[:, 1:2, :, :]

        # Stage 1: Predict flow1
        flow1 = self.stage1_model(x)  # Input is the original image pair

        # Warping layer: Warp image2 using flow1
        # Create grid for warping
        batch_size, _, height, width = image1.size()
        grid = self._create_meshgrid(batch_size, height, width, device=x.device)

        # Normalize flow to grid scale (-1 to 1) - Important for F.grid_sample
        flow1_normalized_x = flow1[:, 0, :, :] / (width / 2)
        flow1_normalized_y = flow1[:, 1, :, :] / (height / 2)
        flow1_normalized = torch.stack(
            (flow1_normalized_x, flow1_normalized_y), dim=1
        )  # N, 2, H, W
        flow1_normalized = flow1_normalized.permute(
            0, 2, 3, 1
        )  # N, H, W, 2 - channels last for grid_sample

        warped_image2_1 = F.grid_sample(
            image2,
            grid + flow1_normalized,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Stage 2: Predict flow2 (residual flow) - Input is image1 and warped_image2_1
        stage2_input = torch.cat(
            (image1, warped_image2_1), dim=1
        )  # Concatenate image1 and warped_image2_1 for stage 2 input
        flow2 = self.stage2_model(stage2_input)  # Model 2 predicts residual flow

        # Combine flows: Simple additive combination for now
        final_flow = flow1 + flow2  # Add flow2 (residual) to flow1

        return final_flow

    def _create_meshgrid(
        self, batch_size, height, width, device
    ):  # Helper function for meshgrid
        x_grid = torch.linspace(-1.0, 1.0, width, device=device)
        y_grid = torch.linspace(-1.0, 1.0, height, device=device)

        x_mesh, y_mesh = torch.meshgrid(
            x_grid, y_grid, indexing="ij"
        )  # Use indexing='ij' for consistent xy ordering

        # Stack and repeat for batch size
        meshgrid = torch.stack((x_mesh, y_mesh), dim=0).float()  # 2, H, W
        meshgrid = meshgrid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # N, 2, H, W
        return meshgrid.permute(
            0, 2, 3, 1
        )  # N, H, W, 2 - channels last for grid_sample


model = MotionVectorRegressionNetworkWithWarping(
    input_images=2, max_displacement=20
).to(device)
if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
print(model)


# %%
def custom_loss(predicted_vectors, target_vectors):
    # l1_loss = nn.functional.l1_loss(predicted_vectors, target_vectors)
    # return l1_loss

    squared_difference = (predicted_vectors - target_vectors) ** 2

    epe_map = torch.sqrt(squared_difference.sum(dim=1))  # N, H, W - EPE for each pixel

    # Average EPE over all pixels and batch
    epe_loss = epe_map.mean()

    return epe_loss


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
    "architecture": "MotionVectorCorrelationRegressionNetworkWithWarping",
    "dataset": {
        "train": len(training_dataset),
        "val": len(validation_dataset),
    },
    "loss_function": "EPE",
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
    current_epoch = epoch

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

            tiles = ["a21_tile_0", "a53_tile_3", "b22_tile_1"]
            for tile in tiles:
                image1_path = GROUPED_TILE_PATHS[tile][10]
                image2_path = GROUPED_TILE_PATHS[tile][11]

                image1 = np.array(Image.open(image1_path))
                image2 = np.array(Image.open(image2_path))

                with torch.no_grad():
                    X = torch.from_numpy(np.array([image1, image2])).float()
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
                            f"tests/sample_{tile}": wandb.Image(
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
