# %% [markdown]
# ### Imports

#  This model adds some warping thing to epe loss model
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
import math
from PIL import Image
from skimage.util import img_as_float
from skimage.restoration import denoise_wavelet
import wandb

import torch
from torch import nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from einops import rearrange
from torch import einsum
import timm
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import OneCycleLR

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
MAX_TILES = 17556  # For running all the images
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
BATCH_SIZE = 8
# BATCH_SIZE = 240  # Fills 32 GB VRAM
IMG_SIZE = TILE_SIZE
LEARNING_RATE = 25e-5
SAVE_FREQUENCY = 1  # Writes a checkpoint file

# Model name for saving files and in wandb
if len(sys.argv) < 2:
    MODEL_NAME = "b4-unknown-test"
else:
    MODEL_NAME = sys.argv[1]
MODEL_FILE = f"{MODEL_NAME}.pth"

if not os.path.exists(MODEL_NAME):
    os.makedirs(MODEL_NAME)

# %% [markdown]
# # Dataset


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

        global current_epoch

        # Get the image index
        path_index = index % NUM_TILES
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

        image = np.array(Image.open(TILE_IMAGE_PATHS[path_index], mode="r"))

        dU, dV = final_field

        new_x = self.pos_x - dU
        new_y = self.pos_y - dV

        denoised_image, extracted_noise = extract_wavelet_noise(img_as_float(image))
        warped_image = map_coordinates(
            denoised_image,
            [new_y, new_x],
            order=0,
            mode="wrap",
        )
        applied_denoised_image = np.clip(warped_image + extracted_noise, 0, 1)

        return np.array([image, applied_denoised_image]).astype(np.float32), np.array(
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


# %% Helper Functions for the model
def bilinear_sampler(img, coords, mode="bilinear"):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


# %% Classes for the model's parts


class Aggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads=4,
        dim_head=128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, "b (h d) x y -> b h (x y) d", h=heads)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


class twins_svt_large(nn.Module):
    def __init__(self, pretrained=True, del_layers=True, in_chans=3):  # Added in_chans
        super().__init__()
        self.svt = timm.create_model(
            "twins_svt_large", pretrained=pretrained, in_chans=in_chans
        )  # Pass in_chans

        if del_layers:
            del self.svt.head
            del self.svt.patch_embeds[2]
            del self.svt.patch_embeds[2]
            del self.svt.blocks[2]
            del self.svt.blocks[2]
            del self.svt.pos_block[2]
            del self.svt.pos_block[2]

    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(
                self.svt.patch_embeds,
                self.svt.pos_drops,
                self.svt.blocks,
                self.svt.pos_block,
            )
        ):

            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == 0:
                x_16 = x.clone()
            if i == layer - 1:
                break

        return x

    def extract_ml_features(self, x, data=None, layer=2):
        res = []
        B = x.shape[0]
        x1 = None

        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(
                self.svt.patch_embeds,
                self.svt.pos_drops,
                self.svt.blocks,
                self.svt.pos_block,
            )
        ):
            x, size = embed(x)
            if i == layer - 1:
                x1 = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer - 1:
                break

        return x1, x

    def compute_params(self):
        num = 0

        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(
                self.svt.patch_embeds,
                self.svt.pos_drops,
                self.svt.blocks,
                self.svt.pos_block,
            )
        ):

            for param in embed.parameters():
                num += np.prod(param.size())
            for param in blocks.parameters():
                num += np.prod(param.size())
            for param in pos_blk.parameters():
                num += np.prod(param.size())
            for param in drop.parameters():
                num += np.prod(param.size())
            if i == 1:
                break
        return num


class Attention(nn.Module):
    def __init__(
        self,
        *,
        args,
        dim,
        max_pos_size=100,
        heads=4,
        dim_head=128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, "b (h d) x y -> b h x y d", h=heads), (q, k))
        q = self.scale * q

        sim = einsum("b h x y d, b h u v d -> b h x y u v", q, k)

        sim = rearrange(sim, "b h x y u v -> b h (x y) (u v)")
        attn = sim.softmax(dim=-1)

        return attn


class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    C_in, C_in, kernel, stride=1, padding=kernel // 2, groups=C_in
                )
                for kernel in k_conv
            ]
        )

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5 * C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5 * C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x


class velocity_update_block(nn.Module):
    def __init__(self, C_in=43 + 128 + 43, C_out=43, C_hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(C_in, C_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(C_hidden, C_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(C_hidden, C_out, 3, padding=1),
        )

    def forward(self, x):
        return self.mlp(x)


class SKMotionEncoder6_Deep_nopool_res(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cor_planes = cor_planes = (
            (args.corr_radius * 2 + 1) ** 2 * args.cost_heads_num * args.corr_levels
        )
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 128, k_conv=args.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=args.k_conv)

        self.init_hidden_state = nn.Parameter(torch.randn(1, 1, 48, 1, 1))

        self.convf1_ = nn.Conv2d(
            4, 128, 1, 1, 0
        )  # No change needed as input is flow (2 channels) concatenated twice
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=args.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(
            64 + 192 + 48 * 3, 128 - 4 + 48, k_conv=args.k_conv
        )

        self.velocity_update_block = velocity_update_block()

    def sample_flo_feat(self, flow, feat):

        sampled_feat = bilinear_sampler(feat.float(), flow.permute(0, 2, 3, 1))
        return sampled_feat

    def forward(
        self,
        motion_hidden_state,
        forward_flow,
        backward_flow,
        coords0,
        forward_corr,
        backward_corr,
        bs,
    ):

        BN, _, H, W = forward_flow.shape
        N = BN // bs

        if motion_hidden_state is None:
            # print("initialized as None")
            motion_hidden_state = self.init_hidden_state.repeat(bs, N, 1, H, W)
        else:
            # print("later iterations")
            motion_hidden_state = motion_hidden_state.reshape(bs, N, -1, H, W)

        forward_loc = forward_flow + coords0
        backward_loc = backward_flow + coords0

        forward_motion_hidden_state = torch.cat(
            [
                motion_hidden_state[:, 1:, ...],
                torch.zeros(bs, 1, 48, H, W).to(motion_hidden_state.device),
            ],
            dim=1,
        ).reshape(BN, -1, H, W)
        forward_motion_hidden_state = self.sample_flo_feat(
            forward_loc, forward_motion_hidden_state
        )
        backward_motion_hidden_state = torch.cat(
            [
                torch.zeros(bs, 1, 48, H, W).to(motion_hidden_state.device),
                motion_hidden_state[:, : N - 1, ...],
            ],
            dim=1,
        ).reshape(BN, -1, H, W)
        backward_motion_hidden_state = self.sample_flo_feat(
            backward_loc, backward_motion_hidden_state
        )

        forward_cor = self.convc1(forward_corr)
        backward_cor = self.convc1(backward_corr)
        cor = F.gelu(torch.cat([forward_cor, backward_cor], dim=1))
        cor: Tensor = self.convc2(cor)

        flow = torch.cat([forward_flow, backward_flow], dim=1)
        flo = self.convf1_(flow)
        flo: Tensor = self.convf2(flo)

        cor_flo = torch.cat(
            [
                cor,
                flo,
                forward_motion_hidden_state,
                backward_motion_hidden_state,
                motion_hidden_state.reshape(BN, -1, H, W),
            ],
            dim=1,
        )
        out = self.conv(cor_flo)

        out, motion_hidden_state = torch.split(out, [124, 48], dim=1)

        return torch.cat([out, flow], dim=1), motion_hidden_state


class SKUpdateBlock6_Deep_nopoolres_AllDecoder2(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.args = args

        args.k_conv = [1, 15]
        args.PCUpdater_conv = [1, 7]

        hidden_dim_ratio = 256 // args.feat_dim

        self.encoder = SKMotionEncoder6_Deep_nopool_res(args)
        self.gru = PCBlock4_Deep_nopool_res(
            128 + hidden_dim + hidden_dim + 128,
            128 // hidden_dim_ratio,
            k_conv=args.PCUpdater_conv,
        )
        self.flow_head = PCBlock4_Deep_nopool_res(
            128 // hidden_dim_ratio, 4, k_conv=args.k_conv
        )  # No change needed as output is flow (2 channels) concatenated twice

        self.mask = nn.Sequential(
            nn.Conv2d(128 // hidden_dim_ratio, 256 // hidden_dim_ratio, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256 // hidden_dim_ratio, args.down_ratio**2 * 9 * 2, 1, padding=0
            ),
        )

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(
        self,
        net,
        motion_hidden_state,
        inp,
        forward_corr,
        backward_corr,
        forward_flow,
        backward_flow,
        coords0,
        attention,
        bs,
    ):

        motion_features, motion_hidden_state = self.encoder(
            motion_hidden_state,
            forward_flow,
            backward_flow,
            coords0,
            forward_corr,
            backward_corr,
            bs=bs,
        )
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 100.0 * self.mask(net)
        return net, motion_hidden_state, mask, delta_flow


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), dim=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


# %% The Model
class MOFNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.hidden_dim = hdim = self.cfg.feat_dim // 2
        self.context_dim = cdim = self.cfg.feat_dim // 2

        cfg.corr_radius = 4

        # feature network, context network, and update block
        self.cnet = twins_svt_large(
            pretrained=self.cfg.pretrain, in_chans=1
        )  # in_chans=1 for grayscale
        self.fnet = twins_svt_large(
            pretrained=self.cfg.pretrain, in_chans=1
        )  # in_chans=1 for grayscale

        hidden_dim_ratio = 256 // cfg.feat_dim

        self.cfg.cost_heads_num = 1
        self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(
            args=self.cfg, hidden_dim=128 // hidden_dim_ratio
        )

        gma_down_ratio = 256 // cfg.feat_dim

        self.att = Attention(
            args=self.cfg,
            dim=128 // hidden_dim_ratio,
            heads=1,
            max_pos_size=160,
            dim_head=128 // hidden_dim_ratio,
        )

    def initialize_flow(self, img, bs, down_ratio):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(bs, H // down_ratio, W // down_ratio).to(img.device)
        coords1 = coords_grid(bs, H // down_ratio, W // down_ratio).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, 3, padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def upsample_flow_4x(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, 3, padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4 * H, 4 * W)

    def upsample_flow_2x(self, flow, mask):

        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, 3, padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 2 * H, 2 * W)

    def forward(self, images, data={}, flow_init=None):

        down_ratio = self.cfg.down_ratio

        B, N, _, H, W = images.shape

        images = 2 * (images / 255.0) - 1.0

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(device_type="cuda", enabled=self.cfg.mixed_precision):
            fmaps = self.fnet(images.reshape(B * N, 1, H, W)).reshape(
                B, N, -1, H // down_ratio, W // down_ratio
            )  # Changed 3 to 1
        fmaps = fmaps.float()

        forward_corr_fn = CorrBlock(
            fmaps[:, 1 : N - 1, ...].reshape(
                B * (N - 2), -1, H // down_ratio, W // down_ratio
            ),
            fmaps[:, 2:N, ...].reshape(
                B * (N - 2), -1, H // down_ratio, W // down_ratio
            ),
            num_levels=self.cfg.corr_levels,
            radius=self.cfg.corr_radius,
        )
        backward_corr_fn = CorrBlock(
            fmaps[:, 1 : N - 1, ...].reshape(
                B * (N - 2), -1, H // down_ratio, W // down_ratio
            ),
            fmaps[:, 0 : N - 2, ...].reshape(
                B * (N - 2), -1, H // down_ratio, W // down_ratio
            ),
            num_levels=self.cfg.corr_levels,
            radius=self.cfg.corr_radius,
        )

        with autocast(device_type="cuda", enabled=self.cfg.mixed_precision):
            cnet = self.cnet(
                images[:, 1 : N - 1, ...].reshape(B * (N - 2), 1, H, W)
            )  # Changed 3 to 1
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            attention = self.att(inp)

        forward_coords1, forward_coords0 = self.initialize_flow(
            images[:, 0, ...], bs=B * (N - 2), down_ratio=down_ratio
        )
        backward_coords1, backward_coords0 = self.initialize_flow(
            images[:, 0, ...], bs=B * (N - 2), down_ratio=down_ratio
        )

        flow_predictions = []  # forward flows followed by backward flows

        motion_hidden_state = None

        for itr in range(self.cfg.decoder_depth):

            forward_coords1 = forward_coords1.detach()
            backward_coords1 = backward_coords1.detach()

            forward_corr = forward_corr_fn(forward_coords1)
            backward_corr = backward_corr_fn(backward_coords1)

            forward_flow = forward_coords1 - forward_coords0
            backward_flow = backward_coords1 - backward_coords0

            with autocast(device_type="cuda", enabled=self.cfg.mixed_precision):
                net, motion_hidden_state, up_mask, delta_flow = self.update_block(
                    net,
                    motion_hidden_state,
                    inp,
                    forward_corr,
                    backward_corr,
                    forward_flow,
                    backward_flow,
                    forward_coords0,
                    attention,
                    bs=B,
                )

            forward_up_mask, backward_up_mask = torch.split(
                up_mask, [down_ratio**2 * 9, down_ratio**2 * 9], dim=1
            )

            forward_coords1 = forward_coords1 + delta_flow[:, 0:2, ...]
            backward_coords1 = backward_coords1 + delta_flow[:, 2:4, ...]

            # upsample predictions
            if down_ratio == 4:
                forward_flow_up = self.upsample_flow_4x(
                    forward_coords1 - forward_coords0, forward_up_mask
                )
                backward_flow_up = self.upsample_flow_4x(
                    backward_coords1 - backward_coords0, backward_up_mask
                )
            elif down_ratio == 2:
                forward_flow_up = self.upsample_flow_2x(
                    forward_coords1 - forward_coords0, forward_up_mask
                )
                backward_flow_up = self.upsample_flow_2x(
                    backward_coords1 - backward_coords0, backward_up_mask
                )
            else:
                forward_flow_up = self.upsample_flow(
                    forward_coords1 - forward_coords0, forward_up_mask
                )
                backward_flow_up = self.upsample_flow(
                    backward_coords1 - backward_coords0, backward_up_mask
                )

            flow_predictions.append(
                torch.cat(
                    [
                        forward_flow_up.reshape(B, N - 2, 2, H, W),
                        backward_flow_up.reshape(B, N - 2, 2, H, W),
                    ],
                    dim=1,
                )
            )

        if self.training:
            return flow_predictions
        else:
            return flow_predictions[-1], flow_predictions[-1]


# %%


class Config:
    def __init__(self):
        self.feat_dim = 256
        self.corr_levels = 4
        self.corr_radius = 4
        self.corr_fn = "default"
        self.mixed_precision = True
        self.decoder_depth = 12
        self.down_ratio = 8
        self.pretrain = True  # for using pretrained model or not


cfg = Config()
model = MOFNet(cfg).to(device)

if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
print(model)


# %%
def sequence_loss(flow_preds, flow_gt) -> Tuple[float, Tensor]:
    """Loss function defined over sequence of flow predictions"""

    # print(flow_gt.shape, valid.shape, flow_preds[0].shape)
    # exit()

    gamma = 0.8
    max_flow = 400
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    B, N, _, H, W = flow_gt.shape

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=2).sqrt()
    valid = mag < max_flow

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        flow_pre = flow_preds[i]
        i_loss = (flow_pre - flow_gt).abs()

        _valid = valid[:, :, None]

        flow_loss += i_weight * (_valid * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=2).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    return flow_loss, epe


# %%
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4, eps=1e-8
)
scheduler = OneCycleLR(
    optimizer,
    LEARNING_RATE,
    125000,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy="linear",
)
scaler = torch.amp.grad_scaler.GradScaler(device="cuda", enabled=True)

# %%
wandb_config = {
    "gpu": GPU,
    "gpu_memory": GPU_MEMORY,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "architecture": "MOFNet",
    "dataset": {
        "train": len(training_dataset),
        "val": len(validation_dataset),
    },
    "loss_function": "EPE",
    "optimizer": "AdamW",
    "scheduler": "OneCycleLR",
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
        loss, metrics, NAN_flag = sequence_loss(pred, batch_vectors)

        scaler.scale(Tensor(loss)).backward()
        scaler.unscale_(optimizer)
        total_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        wandb.log(
            {
                "batch/train_loss": loss,
                "batch/gradient_norm": total_grads,
                "batch/learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        epoch_training_losses.append(loss)

        if batch > (milestone * 100):
            milestone += 1
            print(f"loss: { loss:>7f}  [{batch:>5d}/{size:>5d}]")

    model.eval()
    validation_losses = []

    with torch.no_grad():
        for batch, (batch_images, batch_vectors) in enumerate(validation_dataloader):
            batch_images, batch_vectors = batch_images.to(device), batch_vectors.to(
                device
            )

            pred = model(batch_images)
            loss, metrics, nan = sequence_loss(pred, batch_vectors)
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
            model_artifact = wandb.Artifact(
                name=f"snapshot_{run.id}",
                type="model",
                description="Snapshot at epoch {}".format(epoch),
            )
            model_artifact.add_file("snapshot_save.pt")
            wandb.log_artifact(model_artifact)

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
