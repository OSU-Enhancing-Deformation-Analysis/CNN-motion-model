import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, TypeAlias
import inspect
from scipy.ndimage import map_coordinates
import random
from functools import wraps
import numpy.typing as npt

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


# @vector_field()
def perlin_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    scaled_X = X / 2
    scaled_Y = Y / 2

    # Generate separate Perlin noise for X and Y components
    u = perlin(scaled_X, scaled_Y, seed=0) - 0.5
    v = perlin(scaled_Y, scaled_X, seed=1) - 0.5
    return u, v


# @vector_field()
def noise_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return np.random.normal(0, 0.5, size=(X.shape[0], X.shape[1])).astype(
        np.float32
    ), np.random.normal(0, 0.5, size=(Y.shape[0], Y.shape[1])).astype(np.float32)


def perlin(x: farray, y: farray, seed: int = 0) -> farray:
    # permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype=np.int32)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = np.floor(x), np.floor(y)
    # internal coordinates
    xf, yf = (x - xi).astype(np.float32), (y - yi).astype(np.float32)
    # fade factors
    u, v = perlin_fade(xf), perlin_fade(yf)
    # noise components
    n00 = perlin_gradient(p[p[xi] + yi], xf, yf)
    n01 = perlin_gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = perlin_gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = perlin_gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = perlin_lerp(n00, n10, u)
    x2 = perlin_lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return perlin_lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here


def perlin_lerp(a: farray, b: farray, x: farray) -> farray:
    "linear interpolation"
    return a + x * (b - a)


def perlin_fade(t: farray) -> farray:
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def perlin_gradient(h: npt.NDArray[np.int32], x: farray, y: farray) -> farray:
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


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
