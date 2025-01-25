import os
import glob
import scipy
import random
import numpy as np
import numpy.typing as npt
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from displacements import VectorFieldComposer, VectorField, VECTOR_FIELDS
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
from matplotlib.text import Text
from typing import Callable, List, Tuple, Dict, TypeAlias

IMAGE_SIZE = 256
farray: TypeAlias = npt.NDArray[np.float32]


def load_image_from_tiles(directory="../tiles/") -> npt.NDArray[np.uint8]:
    image_files = glob.glob(os.path.join(directory, "**/*.png"), recursive=True)
    image_path = image_files[random.randint(0, len(image_files) - 1)]
    img = Image.open(image_path, mode="r")
    return np.array(img)


fig = plt.figure(layout="constrained")
gs = GridSpec(3, 6, figure=fig)

vector_ax = fig.add_subplot(gs[0:2, 1:3])
image_ax = fig.add_subplot(gs[0:2, 3:5])
radio_ax = fig.add_subplot(gs[0:4, 0])
slider_ax = fig.add_subplot(gs[2, 1:3])
button_ax = fig.add_subplot(gs[2, 3])
minivector_ax = fig.add_subplot(gs[2, 4:5])
compositions_ax = fig.add_subplot(gs[:, 5])

grid_X, grid_Y = np.meshgrid(
    np.linspace(-1, 1, IMAGE_SIZE), np.linspace(-1, 1, IMAGE_SIZE)
)
minigrid_X, minigrid_Y = np.meshgrid(
    np.linspace(-1, 1, IMAGE_SIZE // 10),
    np.linspace(-1, 1, IMAGE_SIZE // 10),
)

fields = list(VECTOR_FIELDS.values())
field_composer = VectorFieldComposer()

dummy = np.zeros_like(grid_X)
dummy[0, 0] = 1.0
pcolor = vector_ax.pcolormesh(
    grid_X, grid_Y, dummy, cmap="hsv", shading="auto", alpha=0.2
)
quiver = vector_ax.quiver(
    grid_X, grid_Y, dummy, dummy, dummy, cmap="viridis", clim=[0, 8.0]
)
vector_ax.grid(True)
vector_ax.axis("equal")

dummy = np.zeros_like(minigrid_X)
dummy[0, 0] = 1.0
minipcolor = minivector_ax.pcolormesh(
    minigrid_X, minigrid_Y, dummy, cmap="hsv", shading="auto", alpha=0.2
)
miniquiver = minivector_ax.quiver(
    minigrid_X, minigrid_Y, dummy, dummy, dummy, cmap="viridis", clim=[0, 0.8]
)
minivector_ax.grid(True)
minivector_ax.axis("equal")


tile_image = load_image_from_tiles()
display_image = tile_image
image_ax.imshow(display_image)
image_ax.axis("off")

scale_slider = Slider(
    ax=slider_ax,
    label="Vector Scale",
    valmin=0.1,
    valmax=10.0,
    valinit=1.0,
)

button_ax.axis("off")
randomize_button = Button(button_ax.inset_axes((0.1, 0.7, 0.8, 0.2)), label="Randomize")
push_button = Button(button_ax.inset_axes((0.1, 0.4, 0.8, 0.2)), label="Push")
pop_button = Button(button_ax.inset_axes((0.1, 0.1, 0.8, 0.2)), label="Pop")


radio = RadioButtons(radio_ax, [f.__name__ for f in fields], active=1)

field_list_textbox = compositions_ax.text(
    0.1, 0.5, "", transform=compositions_ax.transAxes, fontsize=10
)


def update_field(label):
    new_field = next(f for f in fields if f.__name__ == label)
    field_composer.last().field_func = new_field
    display_transformed_image()

    vector_ax.set_title(label)
    fig.canvas.draw_idle()


def update_scale(val):
    field_composer.last().amplitude = val
    display_transformed_image()
    fig.canvas.draw_idle()


def update_button(val):
    field_composer.fields[-1].randomize()
    display_transformed_image()
    fig.canvas.draw_idle()


def push_field(val):
    field_composer.add_field(field_type="translation_field")
    radio.set_active(0)
    display_transformed_image()
    fig.canvas.draw_idle()


def pop_field(val):
    field_composer.pop_field()
    fig.canvas.draw_idle()


def display_transformed_image():
    global display_image, image_ax, quiver, pcolor

    dU, dV = field_composer.compute_combined_field(grid_X, grid_Y)
    display_image = apply_to_image(tile_image)
    image_ax.imshow(display_image)

    angle = np.arctan2(dV, dU)
    hue = (angle + np.pi) / (2 * np.pi)

    quiver.set_UVC(dU, dV, np.sqrt(dU**2 + dV**2))
    pcolor.set_array(hue.ravel())

    miniU, miniV = field_composer.compute_combined_field(minigrid_X, minigrid_Y)
    miniangle = np.arctan2(miniV, miniU)
    minihue = (miniangle + np.pi) / (2 * np.pi)

    miniquiver.set_UVC(miniU / 10, miniV / 10, np.sqrt(miniU**2 + miniV**2) / 10)
    minipcolor.set_array(minihue.ravel())

    field_list_textbox.set_text(
        "\n".join(field.name for field in field_composer.fields)
    )

    fig.canvas.draw_idle()


def apply_to_image(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    dU, dV = field_composer.compute_combined_field(grid_X, grid_Y)
    x, y = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))

    new_x = x - dU
    new_y = y - dV

    warped_image = np.zeros_like(image)
    warped_image = scipy.ndimage.map_coordinates(
        image, [new_y, new_x], order=1, mode="wrap"
    )

    return warped_image


display_transformed_image()

radio.on_clicked(update_field)
scale_slider.on_changed(update_scale)
randomize_button.on_clicked(update_button)
push_button.on_clicked(push_field)
pop_button.on_clicked(pop_field)
plt.show()
