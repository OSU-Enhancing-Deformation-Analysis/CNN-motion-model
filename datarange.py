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
from tqdm import tqdm  # Import tqdm for the progress bar

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# %% [markdown]
# ### Constants

# %%

TILES_DIR = "../tiles"
TILE_IMAGE_PATHS = glob.glob("../tiles/g*/**/*.png", recursive=True)
NUM_TILES = len(TILE_IMAGE_PATHS)
# Load all images (both stem and graphite)
TILE_SIZE = 256


# %%
from dataclasses import dataclass
from scipy.spatial import (
    Voronoi,
    voronoi_plot_2d,
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
    return (np.sin(innerX) * np.cos(innerY)), (-np.cos(innerX) * np.sin(innerY))


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
    dx = magnitude * np.cos(angle + radius)
    dy = magnitude * np.sin(angle + radius)

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
        # self.center = (0, 0)
        # self.scale = 1.0
        # self.rotation = 0.0
        # self.amplitude = 1.0

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
        meta = {}
        # --- Base field composition ---
        self.composer.clear()
        num_fields = random.randint(1, 2)
        meta["base_fields"] = []
        for _ in range(num_fields):
            field_type = random.choice(self.available_fields)
            meta["base_fields"].append(field_type)
            self.composer.add_field(field_type, randomize=True)
        computed_field = np.array(self.composer.compute_combined_field())

        # --- New Variation Branch ---
        if random.random() > 0.5:
            meta["shape_morph"] = True
            # Shape layer and possible mask inversion
            shape_function = random.choice(shape_functions)
            meta["shape_function"] = shape_function.__name__
            shape_layer = shape_function(TILE_SIZE)  # Image as uint8 0-255

            if random.random() > 0.5:
                meta["mask_inversion"] = True
                original_image = 255 - shape_layer  # Invert mask
            else:
                meta["mask_inversion"] = False

            # Morph the image
            self.composer.clear()
            shape_morph_composer = self.composer
            num_fields = random.randint(1, 2)
            meta["morph_fields"] = []
            for _ in range(num_fields):
                field_type = random.choice(self.available_fields)
                meta["morph_fields"].append(field_type)
                shape_morph_composer.add_field(field_type, randomize=True)

            morphed_shape, _field = shape_morph_composer.apply_to_image(shape_layer)
            if random.random() > 0.3:
                meta["gaussian_blur"] = True
                morphed_shape = gaussian_filter(morphed_shape, sigma=1)
                morphed_shape = (morphed_shape * (255 / np.max(morphed_shape))).astype(
                    np.uint8
                )
            else:
                meta["gaussian_blur"] = False
            morphed_shape = morphed_shape.astype(np.float32) / 255

            if random.random() > 0.5:
                meta["inner_inversion"] = True
                final_field = (1 - morphed_shape) * (
                    computed_field * -1
                ) + morphed_shape * computed_field
            else:
                meta["inner_inversion"] = False
                another_vector_field = VectorFieldComposer()
                num_fields = random.randint(1, 2)
                meta["inner_additional_fields"] = []
                for _ in range(num_fields):
                    field_type = random.choice(self.available_fields)
                    meta["inner_additional_fields"].append(field_type)
                    another_vector_field.add_field(field_type, randomize=True)
                another_computed_field = np.array(
                    another_vector_field.compute_combined_field()
                )
                final_field = (
                    1 - morphed_shape
                ) * computed_field + morphed_shape * another_computed_field
        else:
            meta["shape_morph"] = False
            final_field = computed_field

        image = np.array(Image.open(TILE_IMAGE_PATHS[index % NUM_TILES], mode="r"))
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

        # Now return meta info along with the usual outputs.
        return (
            np.array([image, applied_denoised_image]).astype(np.float32),
            np.array([dU, dV]).astype(np.float32),
            meta,
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

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

# -------------------------------
# Parameters & Data Collection
# -------------------------------
num_samples = 1000  # number of samples to process for stats

# Initialize the dataset
dataset = CustomDataset()
print("Total samples in dataset:", len(dataset))

# Containers to store statistics
original_images = []  # original images
processed_images = []  # processed images
dU_list = []  # vector field component dU for each sample
dV_list = []  # vector field component dV for each sample

# Containers for meta information
meta_info_list = []

# Iterate over the first num_samples items
for i in tqdm(range(num_samples), desc="Processing samples", unit="sample"):
    # Each __getitem__ now returns (images, vector_field, meta)
    (images, vector_field, meta) = dataset[i]
    original, processed = images[0], images[1]
    dU, dV = vector_field[0], vector_field[1]

    original_images.append(original)
    processed_images.append(processed)
    dU_list.append(dU)
    dV_list.append(dV)
    meta_info_list.append(meta)

# Convert lists to numpy arrays for easier processing
original_images = np.array(original_images)  # shape: (num_samples, H, W)
processed_images = np.array(processed_images)  # shape: (num_samples, H, W)
dU_list = np.array(dU_list)  # shape: (num_samples, H, W)
dV_list = np.array(dV_list)  # shape: (num_samples, H, W)


# -------------------------------
# 1. Display a Grid of Random Samples
# -------------------------------
def plot_random_grid(orig_imgs, proc_imgs, n=5):
    """Plot n random samples (original and processed) in a grid."""
    fig, axs = plt.subplots(n, 2, figsize=(10, 2.5 * n))
    indices = np.random.choice(len(orig_imgs), n, replace=False)

    for i, idx in enumerate(indices):
        axs[i, 0].imshow(orig_imgs[idx], cmap="gray")
        axs[i, 0].axis("off")
        axs[i, 0].set_title("Original")

        axs[i, 1].imshow(proc_imgs[idx], cmap="gray")
        axs[i, 1].axis("off")
        axs[i, 1].set_title("Processed")

    plt.tight_layout()
    plt.show()


plot_random_grid(original_images, processed_images, n=5)

# -------------------------------
# 2. Pixel Intensity Distributions
# -------------------------------
orig_pixels = original_images.flatten()
proc_pixels = processed_images.flatten()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(orig_pixels, bins=50, color="blue", alpha=0.7)
plt.title("Original Image Pixel Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(proc_pixels, bins=50, color="green", alpha=0.7)
plt.title("Processed Image Pixel Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.boxplot([orig_pixels, proc_pixels], labels=["Original", "Processed"])
plt.title("Pixel Intensity Box Plots")
plt.ylabel("Pixel Intensity")
plt.show()

diff_images = processed_images - original_images
diff_pixels = diff_images.flatten()
plt.figure(figsize=(6, 4))
plt.hist(diff_pixels, bins=50, color="purple", alpha=0.7)
plt.title("Difference Image Pixel Intensity Distribution")
plt.xlabel("Difference Value")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 3. Vector Field Analysis (dU and dV)
# -------------------------------
dU_pixels = dU_list.flatten()
dV_pixels = dV_list.flatten()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(dU_pixels, bins=50, color="orange", alpha=0.7)
plt.title("dU Component Distribution")
plt.xlabel("dU")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(dV_pixels, bins=50, color="red", alpha=0.7)
plt.title("dV Component Distribution")
plt.xlabel("dV")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.boxplot([dU_pixels, dV_pixels], labels=["dU", "dV"])
plt.title("Box Plots for Vector Field Components")
plt.ylabel("Value")
plt.show()

num_scatter_points = 10000
all_indices = np.arange(dU_pixels.shape[0])
scatter_indices = np.random.choice(all_indices, size=num_scatter_points, replace=False)

plt.figure(figsize=(6, 6))
plt.scatter(
    dU_pixels[scatter_indices], dV_pixels[scatter_indices], alpha=0.3, s=1, color="teal"
)
plt.xlabel("dU")
plt.ylabel("dV")
plt.title("Scatter Plot of dU vs dV")
plt.show()

magnitudes = np.sqrt(dU_pixels**2 + dV_pixels**2)
plt.figure(figsize=(6, 4))
plt.hist(magnitudes, bins=50, color="teal", alpha=0.7)
plt.title("Vector Field Magnitude Distribution")
plt.xlabel("Magnitude")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 4. Spatial Aggregation of Vector Fields
# -------------------------------
mean_dU = np.mean(dU_list, axis=0)
mean_dV = np.mean(dV_list, axis=0)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(mean_dU, cmap="viridis")
plt.colorbar()
plt.title("Average dU over Samples")

plt.subplot(1, 2, 2)
plt.imshow(mean_dV, cmap="viridis")
plt.colorbar()
plt.title("Average dV over Samples")
plt.tight_layout()
plt.show()

var_dU = np.var(dU_list, axis=0)
var_dV = np.var(dV_list, axis=0)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(var_dU, cmap="magma")
plt.colorbar()
plt.title("Variance of dU over Samples")

plt.subplot(1, 2, 2)
plt.imshow(var_dV, cmap="magma")
plt.colorbar()
plt.title("Variance of dV over Samples")
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Quiver Plot Overlay on an Image
# -------------------------------
sample_idx = np.random.randint(0, num_samples)
sample_original = original_images[sample_idx]
sample_dU = dU_list[sample_idx]
sample_dV = dV_list[sample_idx]

H, W = sample_dU.shape
X, Y = np.meshgrid(np.arange(W), np.arange(H))

plt.figure(figsize=(8, 8))
plt.imshow(sample_original, cmap="gray")
plt.quiver(X, Y, sample_dU, sample_dV, color="red", scale=50)
plt.title("Quiver Plot Overlay on Original Image")
plt.axis("off")
plt.show()

# -------------------------------
# 6. Random Conditions & Vector Field Usage Analysis
# -------------------------------

# Prepare containers to collect meta info across samples.
base_fields_all = []
morph_fields_all = []
inner_additional_fields_all = []
shape_morph_flags = []
mask_inversion_flags = []
gaussian_blur_flags = []
inner_inversion_flags = []

for meta in meta_info_list:
    base_fields_all.extend(meta.get("base_fields", []))
    morph_fields_all.extend(meta.get("morph_fields", []))
    inner_additional_fields_all.extend(meta.get("inner_additional_fields", []))
    shape_morph_flags.append(meta.get("shape_morph", False))
    mask_inversion_flags.append(meta.get("mask_inversion", False))
    gaussian_blur_flags.append(meta.get("gaussian_blur", False))
    inner_inversion_flags.append(meta.get("inner_inversion", False))

# Count the frequency of each vector field type in different branches.
base_fields_counter = Counter(base_fields_all)
morph_fields_counter = Counter(morph_fields_all)
inner_additional_fields_counter = Counter(inner_additional_fields_all)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.bar(
    list(base_fields_counter.keys()),
    list(base_fields_counter.values()),
    color="skyblue",
)
plt.title("Base Fields Usage")
plt.xticks(rotation=45)
plt.xlabel("Vector Field Type")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
plt.bar(
    list(morph_fields_counter.keys()),
    list(morph_fields_counter.values()),
    color="salmon",
)
plt.title("Morph Fields Usage")
plt.xticks(rotation=45)
plt.xlabel("Vector Field Type")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
plt.bar(
    list(inner_additional_fields_counter.keys()),
    list(inner_additional_fields_counter.values()),
    color="lightgreen",
)
plt.title("Inner Additional Fields Usage")
plt.xticks(rotation=45)
plt.xlabel("Vector Field Type")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Count how often each random branch was taken.
shape_morph_count = sum(shape_morph_flags)
mask_inversion_count = sum(mask_inversion_flags)
gaussian_blur_count = sum(gaussian_blur_flags)
inner_inversion_count = sum(inner_inversion_flags)
conditions = ["Shape Morph", "Mask Inversion", "Gaussian Blur", "Inner Inversion"]
counts = [
    shape_morph_count,
    mask_inversion_count,
    gaussian_blur_count,
    inner_inversion_count,
]

plt.figure(figsize=(8, 5))
plt.bar(conditions, counts, color=["blue", "orange", "green", "red"])
plt.title("Random Conditions Occurrence Counts")
plt.ylabel("Count")
plt.show()
