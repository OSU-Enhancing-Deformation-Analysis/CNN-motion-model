import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Utility Functions
def create_grid(n=20, domain=(-2, 2)):
    x = np.linspace(domain[0], domain[1], n)
    y = np.linspace(domain[0], domain[1], n)
    X, Y = np.meshgrid(x, y)
    return X, Y


def plot_vector_field(X, Y, U, V, title, density=1):
    plt.figure(figsize=(10, 10))
    density = max(1, int(density))
    plt.quiver(
        X[::density],
        Y[::density],
        U[::density],
        V[::density],
        np.sqrt(U[::density] ** 2 + V[::density] ** 2),
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


# Basic Vector Fields
def translation_field(X, Y):
    return np.ones_like(X), np.zeros_like(Y)


def rotation_field(X, Y):
    return -Y, X


def shear_field(X, Y):
    return Y, np.zeros_like(Y)


def scale_field(X, Y):
    return X, Y


def gradient_field(X, Y):
    scalar_field = X**2 + Y**2
    return np.gradient(scalar_field, axis=0), np.gradient(scalar_field, axis=1)


def curl_field(X, Y):
    return Y, -X


def harmonic_field(X, Y):
    return np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y), -np.cos(
        2 * np.pi * X
    ) * np.sin(2 * np.pi * Y)


def pearling_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return np.sin(2 * np.pi * X) * X / (r + 1e-3), np.cos(2 * np.pi * Y) * Y / (
        r + 1e-3
    )


def outward_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return X / (r + 1e-3), Y / (r + 1e-3)


def compressing_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return -X / (r + 1e-3), -Y / (r + 1e-3)


def point_source(X, Y, x0=0, y0=0, strength=1):
    dx, dy = X - x0, Y - y0
    r = np.sqrt(dx**2 + dy**2)
    return strength * dx / (r**2 + 1e-3), strength * dy / (r**2 + 1e-3)


def point_sink(X, Y, x0=0, y0=0, strength=1):
    dx, dy = X - x0, Y - y0
    r = np.sqrt(dx**2 + dy**2)
    return -strength * dx / (r**2 + 1e-3), -strength * dy / (r**2 + 1e-3)


# Compound and Complex Fields
def compound_field1(X, Y):
    U1, V1 = rotation_field(X, Y)
    U2, V2 = outward_field(X, Y)
    return U1 + 0.5 * U2, V1 + 0.5 * V2


def compound_field2(X, Y):
    U1, V1 = harmonic_field(X, Y)
    U2, V2 = pearling_field(X, Y)
    return U1 + U2, V1 + V2


def compound_field3(X, Y):
    U1, V1 = point_source(X, Y, -1, -1)
    U2, V2 = point_sink(X, Y, 1, 1)
    U3, V3 = rotation_field(X, Y)
    return U1 + U2 + 0.5 * U3, V1 + V2 + 0.5 * V3


def turbulent_field(X, Y):
    sources = [(-1, -1, 1), (1, 1, -1), (0, 0, 0.5)]
    U_total, V_total = np.zeros_like(X), np.zeros_like(Y)
    for x0, y0, strength in sources:
        U, V = point_source(X, Y, x0, y0, strength)
        U_total += U
        V_total += V
    U_vortex, V_vortex = curl_field(X, Y)
    return U_total + 0.5 * U_vortex, V_total + 0.5 * V_vortex


# Time-Dependent Fields
def time_varying_field(X, Y, t):
    return np.cos(X + t) * np.sin(Y), np.sin(X) * np.cos(Y - t)


def rotating_dipole(X, Y, t):
    X_rot = X * np.cos(t) - Y * np.sin(t)
    Y_rot = X * np.sin(t) + Y * np.cos(t)
    U, V = point_source(X_rot, Y_rot)
    return U * np.cos(-t) - V * np.sin(-t), U * np.sin(-t) + V * np.cos(-t)


# Animation


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


# Main Execution
if __name__ == "__main__":
    X, Y = create_grid(30)

    # Define fields
    fields = {
        "Translation": translation_field,
        "Rotation": rotation_field,
        "Shear": shear_field,
        "Scale": scale_field,
        "Gradient": gradient_field,
        "Curl": curl_field,
        "Harmonic": harmonic_field,
        "Pearling": pearling_field,
        "Outward": outward_field,
        "Compressing": compressing_field,
        "Point Source": lambda X, Y: point_source(X, Y, -1, -1),
        "Point Sink": lambda X, Y: point_sink(X, Y, 1, 1),
        "Compound 1": compound_field1,
        "Compound 2": compound_field2,
        "Compound 3": compound_field3,
        "Turbulent": turbulent_field,
    }

    # Plot static fields
    # for name, func in fields.items():
    #     U, V = func(X, Y)
    #     plot_vector_field(X, Y, U, V, name)

    plot_all_fields_grid(fields, grid_size=(4, 4), n=30, domain=(-2, 2))

    # Animate fields
    print("Animating Time-Varying Field...")
    animate_field(time_varying_field)

    print("Animating Rotating Dipole...")
    animate_field(rotating_dipole)
