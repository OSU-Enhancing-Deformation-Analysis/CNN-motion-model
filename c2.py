import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Grid generation
def create_grid(n=20, domain=(-2, 2)):
    x = np.linspace(domain[0], domain[1], n)
    y = np.linspace(domain[0], domain[1], n)
    X, Y = np.meshgrid(x, y)
    return X, Y


# Vector field plotting
def plot_vector_field(X, Y, U, V, title, density=1):
    plt.figure(figsize=(10, 10))
    density = max(1, int(density))  # Ensure density is an integer
    plt.quiver(
        X[::density],
        Y[::density],
        U[::density],
        V[::density],
        np.sqrt(U[::density] ** 2 + V[::density] ** 2),  # Color by magnitude
        cmap="viridis",
    )
    plt.colorbar(label="Magnitude")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_all_fields_grid(fields, grid_size=(4, 4), n=20, domain=(-2, 2)):
    """Plot all vector fields in a grid layout."""
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
    X, Y = create_grid(n, domain)

    for ax, (name, field_func) in zip(axes.flatten(), fields.items()):
        U, V = field_func(X, Y)
        ax.quiver(X, Y, U, V, np.sqrt(U**2 + V**2), cmap="viridis")
        ax.set_title(name)
        ax.axis("equal")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# Basic vector fields
def translation_field(X, Y):
    return np.ones_like(X), np.zeros_like(Y)


def rotation_field(X, Y):
    return -Y, X


def shear_field(X, Y):
    return np.ones_like(X), X


def scale_field(X, Y):
    return X, Y


def gradient_field(X, Y):
    return 2 * X, 2 * Y


def curl_field(X, Y):
    return -Y, X


def harmonic_field(X, Y):
    return np.sin(X), np.cos(Y)


def pearling_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return np.sin(r) * X / r, np.sin(r) * Y / r


def uniform_field(X, Y):
    return np.ones_like(X), np.ones_like(Y)


def outward_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return X / (r + 1e-3), Y / (r + 1e-3)


def compressing_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return -X / (r + 1e-3), -Y / (r + 1e-3)


def point_sink(X, Y, x0=0, y0=0):
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2)
    return -dx / (r**2 + 1e-3), -dy / (r**2 + 1e-3)


def point_source(X, Y, x0=0, y0=0):
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2)
    return dx / (r**2 + 1e-3), dy / (r**2 + 1e-3)


def dipole_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return Y / (2 * np.pi * (r**2 + 1e-3)), -X / (2 * np.pi * (r**2 + 1e-3))


def vortex_field(X, Y, strength=1.0):
    r = np.sqrt(X**2 + Y**2)
    return -strength * Y / (r**2 + 0.1), strength * X / (r**2 + 0.1)


# Compound fields
def combine_fields(field_funcs, weights=None):
    if weights is None:
        weights = [1.0] * len(field_funcs)

    def combined_field(X, Y):
        U_total = np.zeros_like(X)
        V_total = np.zeros_like(Y)
        for field_func, weight in zip(field_funcs, weights):
            U, V = field_func(X, Y)
            U_total += weight * U
            V_total += weight * V
        return U_total, V_total

    return combined_field


# Animated fields
def animate_field(field_func, domain=(-2, 2), n=20, frames=100, interval=50):
    X, Y = create_grid(n, domain)

    fig, ax = plt.subplots(figsize=(10, 10))
    Q = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y))
    ax.grid(True)
    ax.set_aspect("equal")

    def update(frame):
        t = frame * 2 * np.pi / frames
        U, V = field_func(X, Y, t)
        Q.set_UVC(U, V)
        return (Q,)

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.show()
    return anim


# Example usage
if __name__ == "__main__":
    X, Y = create_grid(30)

    # Define and plot basic fields
    vector_fields = {
        "Translation": translation_field,
        "Rotation": rotation_field,
        "Shear": shear_field,
        "Scale": scale_field,
        "Gradient": gradient_field,
        "Curl": curl_field,
        "Harmonic": harmonic_field,
        "Pearling": pearling_field,
        "Uniform": uniform_field,
        "Outward": outward_field,
        "Compressing": compressing_field,
        "Point Sink": point_sink,
        "Point Source": point_source,
        "Dipole": dipole_field,
        "Vortex": vortex_field,
    }

    # for name, field_func in vector_fields.items():
    #     U, V = field_func(X, Y)
    #     plot_vector_field(X, Y, U, V, name)

    plot_all_fields_grid(vector_fields, grid_size=(4, 4), n=30, domain=(-2, 2))

    # Compound and animated fields
    print("Animating rotating dipole...")
    animate_field(
        lambda X, Y, t: dipole_field(
            X * np.cos(t) - Y * np.sin(t), X * np.sin(t) + Y * np.cos(t)
        )
    )
