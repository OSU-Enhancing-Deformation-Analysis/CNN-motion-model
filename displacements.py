import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict
import inspect
from scipy.ndimage import map_coordinates
import random
from functools import wraps

# Global registry for vector fields
VECTOR_FIELDS: Dict[str, Callable] = {}


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
def translation_field(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.ones_like(X), np.zeros_like(Y)


@vector_field()
def rotation_field(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return -Y, X


@vector_field()
def shear_field(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.ones_like(X), X


@vector_field()
def shear_field2(X, Y):
    return Y, np.zeros_like(Y)


@vector_field()
def scale_field(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return X, Y


@vector_field()
def gradient_field(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return X**2, Y**2


@vector_field()
def gradient_field2(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    field = X**2 + Y**2
    return np.gradient(field, axis=0) * 10, np.gradient(field, axis=1) * 10


@vector_field()
def harmonic_field(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.sin(X), np.cos(Y)


@vector_field()
def harmonic_field2(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y), -np.cos(
        2 * np.pi * X
    ) * np.sin(2 * np.pi * Y)


@vector_field()
def pearling_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return np.sin(2 * np.pi * X) * X / (r + 1e-3), np.cos(2 * np.pi * Y) * Y / (
        r + 1e-3
    )


@vector_field()
def outward_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return X / (r + 1e-3), Y / (r + 1e-3)


@vector_field()
def compressing_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return -X / (r + 1e-3), -Y / (r + 1e-3)


@vector_field()
def vortex_field(X, Y, strength=1.0):
    r = np.sqrt(X**2 + Y**2)
    return -strength * Y / (r**2 + 0.1), strength * X / (r**2 + 0.1)


@vector_field()
def perlin_field(X, Y):
    scaled_X = X / 2
    scaled_Y = Y / 2

    # Generate separate Perlin noise for X and Y components
    u = perlin(scaled_X, scaled_Y, seed=0) - 0.5
    v = perlin(scaled_Y, scaled_X, seed=1) - 0.5
    return u, v


@vector_field()
def noise_field(X, Y):
    return np.random.normal(0, 0.5, size=(X.shape[0], X.shape[1])), np.random.normal(
        0, 0.5, size=(Y.shape[0], Y.shape[1])
    )


def perlin(x, y, seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
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


def perlin_lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)


def perlin_fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def perlin_gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
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

    def apply(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, image_size: int):
        self.image_size = image_size
        self.fields: List[VectorField] = []
        self.X, self.Y = np.meshgrid(
            np.linspace(-1, 1, image_size), np.linspace(-1, 1, image_size)
        )

        self.available_fields = dict(VECTOR_FIELDS)

    def add_field(self, field_type: str, **kwargs) -> None:
        """Add a vector field with specified parameters."""
        if field_type not in self.available_fields:
            raise ValueError(f"Unknown field type: {field_type}")

        field = VectorField(
            name=field_type, field_func=self.available_fields[field_type], **kwargs
        )
        self.fields.append(field)

    def clear_fields(self) -> None:
        """Remove all fields."""
        self.fields = []

    def compute_combined_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the combined vector field."""
        total_dx = np.zeros_like(self.X)
        total_dy = np.zeros_like(self.Y)

        for field in self.fields:
            dx, dy = field.apply(self.X, self.Y)
            total_dx += dx
            total_dy += dy

        return total_dx, total_dy

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """Apply the combined field to deform an image."""
        dx, dy = self.compute_combined_field()

        dx = dx * self.image_size / 2
        dy = dy * self.image_size / 2

        y, x = np.meshgrid(
            np.arange(self.image_size), np.arange(self.image_size), indexing="ij"
        )

        coords = np.stack([y + dy, x + dx])

        # Ensure coordinates stay within bounds
        coords[0] = np.clip(coords[0], 0, self.image_size - 1)
        coords[1] = np.clip(coords[1], 0, self.image_size - 1)

        return map_coordinates(image, coords, order=3)
