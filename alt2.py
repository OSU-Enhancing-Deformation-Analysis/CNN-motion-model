import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_grid(n=20, domain=(-2, 2)):
    x = np.linspace(domain[0], domain[1], n)
    y = np.linspace(domain[0], domain[1], n)
    X, Y = np.meshgrid(x, y)
    return X, Y

def plot_vector_field(X, Y, U, V, title, density=1):  # Changed density to default to 1
    plt.figure(figsize=(10, 10))
    # Ensure density is an integer
    density = max(1, int(density))
    plt.quiver(X[::density], Y[::density], 
              U[::density], V[::density],
              np.sqrt(U[::density]**2 + V[::density]**2),  # Color by magnitude
              cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Additional basic fields
def spiral_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    return r*np.cos(theta) - Y, r*np.sin(theta) + X

def vortex_field(X, Y, strength=1.0):
    r = np.sqrt(X**2 + Y**2)
    return -strength*Y/(r**2 + 0.1), strength*X/(r**2 + 0.1)

def saddle_field(X, Y):
    return X, -Y

def dipole_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return Y/(2*np.pi*(r**2)), -X/(2*np.pi*(r**2))

def potential_flow(X, Y, sources):
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for x0, y0, strength in sources:
        dx = X - x0
        dy = Y - y0
        r = np.sqrt(dx**2 + dy**2)
        U += strength * dx / (2*np.pi*r**2)
        V += strength * dy / (2*np.pi*r**2)
    return U, V

# Time-dependent fields
def time_varying_field(X, Y, t):
    return np.cos(X + t)*np.sin(Y), np.sin(X)*np.cos(Y - t)

def rotating_dipole(X, Y, t):
    Xrot = X*np.cos(t) - Y*np.sin(t)
    Yrot = X*np.sin(t) + Y*np.cos(t)
    U, V = dipole_field(Xrot, Yrot)
    return (U*np.cos(-t) - V*np.sin(-t), 
            U*np.sin(-t) + V*np.cos(-t))

# Complex combinations
def turbulent_field(X, Y):
    # Combine vortices, sources, and sinks
    sources = [(-1, -1, 1), (1, 1, -1), (0, 0, 0.5)]
    U1, V1 = potential_flow(X, Y, sources)
    U2, V2 = vortex_field(X, Y, 0.5)
    return U1 + U2, V1 + V2

def wave_interference(X, Y):
    k1, k2 = 2, 3  # Wave numbers
    return (np.sin(k1*X) + np.sin(k2*Y), 
            np.cos(k1*X) - np.cos(k2*Y))

# Animation function
def animate_field(field_func, domain=(-2, 2), n=20, frames=100, interval=50):
    X, Y = create_grid(n, domain)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    Q = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y))
    ax.grid(True)
    ax.set_aspect('equal')
    
    def update(frame):
        t = frame * 2*np.pi / frames
        U, V = field_func(X, Y, t)
        Q.set_UVC(U, V)
        return Q,
    
    anim = FuncAnimation(fig, update, frames=frames, 
                        interval=interval, blit=True)
    plt.show()
    return anim

# Example usage
if __name__ == "__main__":
    X, Y = create_grid(30)
    
    # Plot static fields
    plot_vector_field(X, Y, *spiral_field(X, Y), "Spiral Field")
    plot_vector_field(X, Y, *vortex_field(X, Y), "Vortex Field")
    plot_vector_field(X, Y, *saddle_field(X, Y), "Saddle Field")
    plot_vector_field(X, Y, *dipole_field(X, Y), "Dipole Field")
    plot_vector_field(X, Y, *turbulent_field(X, Y), "Turbulent Field")
    plot_vector_field(X, Y, *wave_interference(X, Y), "Wave Interference")
    
    # Animate time-dependent fields
    print("Animating time-varying field...")
    animate_field(time_varying_field)
    
    print("Animating rotating dipole...")
    animate_field(rotating_dipole)
    
    # Example of potential flow with multiple sources/sinks
    sources = [(-1, -1, 1), (1, 1, -1), (0, 0, 0.5)]
    U, V = potential_flow(X, Y, sources)
    plot_vector_field(X, Y, U, V, "Multiple Sources/Sinks")

# Additional utility functions for field manipulation
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

def scale_field(field_func, scale):
    return lambda X, Y: tuple(scale * x for x in field_func(X, Y))

def rotate_field(field_func, angle):
    def rotated_field(X, Y):
        U, V = field_func(X, Y)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return (U*cos_a - V*sin_a, U*sin_a + V*cos_a)
    return rotated_field