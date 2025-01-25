import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from displacements import VectorFieldComposer, VectorField
from matplotlib.gridspec import GridSpec
import os
from PIL import Image
from scipy.ndimage import map_coordinates
import scipy
import glob
import random


def load_image_from_tiles(
    directory="../tiles/",
):
    # Glob get list of image files in the tiles directory
    image_files = glob.glob(os.path.join(directory, "**/*.png"), recursive=True)
    image_path = image_files[random.randint(0, len(image_files) - 1)]
    img = Image.open(image_path, mode="r")
    return np.array(img)


image_size = 256


fig = plt.figure(layout="constrained")

gs = GridSpec(3, 5, figure=fig)
vector_ax = fig.add_subplot(gs[0:2, 1:3])
image_ax = fig.add_subplot(gs[0:2, 3:5])
radio_ax = fig.add_subplot(gs[0:4, 0])
slider_ax = fig.add_subplot(gs[2, 1:3])
minivector_ax = fig.add_subplot(gs[2, 3:5])

composer = VectorFieldComposer(image_size=image_size)
minicomposer = VectorFieldComposer(image_size=image_size // 10)

fields = list(composer.available_fields.values())
field = fields[1]

U, V = field(composer.X, composer.Y)
magnitude = np.sqrt(U**2 + V**2)
angle = np.arctan2(V, U)
hue = (angle + np.pi) / (2 * np.pi)

pcolor = vector_ax.pcolormesh(
    composer.X, composer.Y, hue, cmap="hsv", shading="auto", alpha=0.2
)
quiver = vector_ax.quiver(composer.X, composer.Y, U, V, magnitude, cmap="viridis")

miniU, miniV = field(minicomposer.X, minicomposer.Y)
minimagnitude = np.sqrt(miniU**2 + miniV**2)
miniangle = np.arctan2(miniV, miniU)
minihue = (miniangle + np.pi) / (2 * np.pi)
minipcolor = minivector_ax.pcolormesh(
    minicomposer.X, minicomposer.Y, minihue, cmap="hsv", shading="auto", alpha=0.2
)
quiver2 = minivector_ax.quiver(
    minicomposer.X, minicomposer.Y, miniU, miniV, minimagnitude, cmap="viridis"
)

vector_ax.set_title(field.__name__)
vector_ax.grid(True)
vector_ax.axis("equal")

scale_slider = Slider(
    ax=slider_ax,
    label="Vector Scale",
    valmin=0.1,
    valmax=10.0,
    valinit=1.0,
)

radio = RadioButtons(radio_ax, [f.__name__ for f in fields])

tile_image = load_image_from_tiles()
display_image = tile_image
image_ax.imshow(display_image)
image_ax.set_title("Tile Image")
image_ax.axis("off")


def update_field(label):
    global field, display_image, image_ax, quiver, pcolor, image_size
    field = next(f for f in fields if f.__name__ == label)
    scale = scale_slider.val
    U, V = field(composer.X, composer.Y)
    angle = np.arctan2(V, U)
    hue = (angle + np.pi) / (2 * np.pi)

    quiver.set_UVC(U * scale, V * scale)
    pcolor.set_array(hue.ravel())

    display_image = apply_to_image(tile_image, U * scale, V * scale)
    image_ax.imshow(display_image)

    miniU, miniV = field(minicomposer.X, minicomposer.Y)
    miniangle = np.arctan2(miniV, miniU)
    minihue = (miniangle + np.pi) / (2 * np.pi)

    quiver2.set_UVC(miniU * scale, miniV * scale)
    minipcolor.set_array(minihue.ravel())

    vector_ax.set_title(label)
    fig.canvas.draw_idle()


def update_scale(val):
    global field, display_image, image_ax, quiver, pcolor, image_size
    U, V = field(composer.X, composer.Y)
    quiver.set_UVC(U * val, V * val)

    display_image = apply_to_image(tile_image, U * val, V * val)
    image_ax.imshow(display_image)

    miniU, miniV = field(minicomposer.X, minicomposer.Y)
    quiver2.set_UVC(miniU * val, miniV * val)

    fig.canvas.draw_idle()


def apply_to_image(image, U, V):

    # Create coordinate grids
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))

    new_x = x - U
    new_y = y - V

    warped_image = np.zeros_like(image)

    warped_image = scipy.ndimage.map_coordinates(
        image, [new_y, new_x], order=1, mode="wrap"
    )

    return warped_image


# def apply_to_image():
#     global field, display_image, image_ax, quiver, pcolor, image_size
#     U, V = field(composer.X, composer.Y)
#     dx = np.cos(U) * V
#     dy = np.sin(U) * V

#     new_display_image = np.zeros_like(display_image)

#     # Loop over every pixel in the image
#     for i in range(image_size):
#         for j in range(image_size):
#             # Calculate the new position of the pixel
#             new_x = i + dx[i, j]
#             new_y = j + dy[i, j]

#             # Round to the nearest pixel
#             new_x = round(new_x)
#             new_y = round(new_y)

#             # Ensure we're within image bounds
#             new_x = max(0, min(new_x, image_size - 1))
#             new_y = max(0, min(new_y, image_size - 1))

#             # Pick the nearest pixel
#             new_display_image[int(new_x), int(new_y)] = tile_image[i, j]

#     # Update the display image
#     display_image[:] = new_display_image


radio.on_clicked(update_field)
scale_slider.on_changed(update_scale)
plt.show()
