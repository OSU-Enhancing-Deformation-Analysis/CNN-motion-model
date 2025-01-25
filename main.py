import os
import glob
import scipy
import random
import numpy as np
import numpy.typing as npt
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from displacements import VectorFieldComposer, VectorField
from matplotlib.widgets import Slider, RadioButtons, Button
from typing import Callable, List, Tuple, Dict, TypeAlias

IMAGE_SIZE = 256
farray: TypeAlias = npt.NDArray[np.float32]


def load_image_from_tiles(directory="../tiles/") -> npt.NDArray[np.uint8]:
    image_files = glob.glob(os.path.join(directory, "**/*.png"), recursive=True)
    image_path = image_files[random.randint(0, len(image_files) - 1)]
    img = Image.open(image_path, mode="r")
    return np.array(img)


fig = plt.figure(layout="constrained")
gs = GridSpec(3, 5, figure=fig)

vector_ax = fig.add_subplot(gs[0:2, 1:3])
image_ax = fig.add_subplot(gs[0:2, 3:5])
radio_ax = fig.add_subplot(gs[0:4, 0])
slider_ax = fig.add_subplot(gs[2, 1:3])
button_ax = fig.add_subplot(gs[2, 3])
minivector_ax = fig.add_subplot(gs[2, 4:5])

composer = VectorFieldComposer(image_size=IMAGE_SIZE)
minicomposer = VectorFieldComposer(image_size=IMAGE_SIZE // 10)

fields = list(composer.available_fields.values())
current_field = fields[1]

field_creator = VectorField(name="Random", field_func=current_field)
field_creator.randomize()

U, V = current_field(composer.X, composer.Y)
magnitude = np.sqrt(U**2 + V**2)
angle = np.arctan2(V, U)
hue = (angle + np.pi) / (2 * np.pi)

pcolor = vector_ax.pcolormesh(
    composer.X, composer.Y, hue, cmap="hsv", shading="auto", alpha=0.2
)
quiver = vector_ax.quiver(composer.X, composer.Y, U, V, magnitude, cmap="viridis")
vector_ax.grid(True)
vector_ax.axis("equal")

miniU, miniV = current_field(minicomposer.X, minicomposer.Y)
minimagnitude = np.sqrt(miniU**2 + miniV**2)
miniangle = np.arctan2(miniV, miniU)
minihue = (miniangle + np.pi) / (2 * np.pi)
minipcolor = minivector_ax.pcolormesh(
    minicomposer.X, minicomposer.Y, minihue, cmap="hsv", shading="auto", alpha=0.2
)
miniquiver = minivector_ax.quiver(
    minicomposer.X, minicomposer.Y, miniU, miniV, minimagnitude, cmap="viridis"
)
minivector_ax.grid(True)
minivector_ax.axis("equal")


tile_image = load_image_from_tiles()
display_image = tile_image
image_ax.imshow(display_image)
image_ax.set_title("Tile Image")
image_ax.axis("off")

scale_slider = Slider(
    ax=slider_ax,
    label="Vector Scale",
    valmin=0.1,
    valmax=10.0,
    valinit=1.0,
)

button_ax.axis("off")
randomize_button = Button(button_ax, label="Randomize")

radio = RadioButtons(radio_ax, [f.__name__ for f in fields], active=1)


def update_field(label):
    global current_field, display_image, image_ax, quiver, pcolor
    current_field = next(f for f in fields if f.__name__ == label)

    display_transformed_image()

    vector_ax.set_title(label)
    fig.canvas.draw_idle()


def update_scale(val):
    global current_field, display_image, image_ax, quiver, pcolor, amplitude_scale

    display_transformed_image()

    fig.canvas.draw_idle()


def update_button(val):
    global current_field, display_image, image_ax, quiver, pcolor, amplitude_scale
    field_creator.randomize()
    display_transformed_image()
    fig.canvas.draw_idle()


def display_transformed_image():
    global current_field, display_image, image_ax, quiver, pcolor
    field_creator.field_func = current_field
    field_creator.amplitude = scale_slider.val

    U, V = field_creator.apply(composer.X, composer.Y)
    display_image = apply_to_image(tile_image, U, V)
    image_ax.imshow(display_image)

    angle = np.arctan2(V, U)
    hue = (angle + np.pi) / (2 * np.pi)

    quiver.set_UVC(U, V)
    pcolor.set_array(hue.ravel())

    miniU, miniV = field_creator.apply(minicomposer.X, minicomposer.Y)
    miniangle = np.arctan2(miniV, miniU)
    minihue = (miniangle + np.pi) / (2 * np.pi)

    miniquiver.set_UVC(miniU / 10, miniV / 10)
    minipcolor.set_array(minihue.ravel())

    vector_ax.set_title(field_creator.name)
    fig.canvas.draw_idle()


def apply_to_image(
    image: npt.NDArray[np.uint8], U: farray, V: farray
) -> npt.NDArray[np.uint8]:
    x, y = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))

    new_x = x - U
    new_y = y - V

    warped_image = np.zeros_like(image)
    warped_image = scipy.ndimage.map_coordinates(
        image, [new_y, new_x], order=1, mode="wrap"
    )

    return warped_image


display_transformed_image()

radio.on_clicked(update_field)
scale_slider.on_changed(update_scale)
randomize_button.on_clicked(update_button)
plt.show()
