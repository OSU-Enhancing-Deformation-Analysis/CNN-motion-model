import numpy as np
import matplotlib.pyplot as plt

def create_grid(n=20):
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    return X, Y

def plot_vector_field(X, Y, U, V, title):
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V)
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
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
    return 2*X, 2*Y

def curl_field(X, Y):
    return -Y, X

def harmonic_field(X, Y):
    return np.sin(X), np.cos(Y)

def pearling_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return np.sin(r)*X/r, np.sin(r)*Y/r

def uniform_field(X, Y):
    return np.ones_like(X), np.ones_like(Y)

def outward_field(X, Y):
    r = np.sqrt(X**2 + Y**2)
    return X/r, Y/r

def point_sink(X, Y, x0=0, y0=0):
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2)
    return -dx/r**2, -dy/r**2

def point_source(X, Y, x0=0, y0=0):
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2)
    return dx/r**2, dy/r**2

def wild_field(X, Y):
    return np.sin(X*Y), np.cos(X+Y)

# Compound fields
def compound_field1(X, Y):
    # Rotation + Outward
    U1, V1 = rotation_field(X, Y)
    U2, V2 = outward_field(X, Y)
    return U1 + 0.5*U2, V1 + 0.5*V2

def compound_field2(X, Y):
    # Harmonic + Pearling
    U1, V1 = harmonic_field(X, Y)
    U2, V2 = pearling_field(X, Y)
    return U1 + U2, V1 + V2

def compound_field3(X, Y):
    # Source + Sink + Rotation
    U1, V1 = point_source(X, Y, -1, -1)
    U2, V2 = point_sink(X, Y, 1, 1)
    U3, V3 = rotation_field(X, Y)
    return U1 + U2 + 0.5*U3, V1 + V2 + 0.5*V3

# Main execution
if __name__ == "__main__":
    X, Y = create_grid(20)
    
    # Plot basic fields
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
        "Point Sink": point_sink,
        "Point Source": point_source,
        "Wild Field": wild_field,
    }
    
    for name, field_func in vector_fields.items():
        U, V = field_func(X, Y)
        plot_vector_field(X, Y, U, V, name)
    
    # Plot compound fields
    compound_fields = {
        "Rotation + Outward": compound_field1,
        "Harmonic + Pearling": compound_field2,
        "Source + Sink + Rotation": compound_field3,
    }
    
    for name, field_func in compound_fields.items():
        U, V = field_func(X, Y)
        plot_vector_field(X, Y, U, V, name)