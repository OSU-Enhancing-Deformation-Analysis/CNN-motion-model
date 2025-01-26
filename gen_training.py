import os
import glob
import random
import numpy as np
import numpy.typing as npt
from PIL import Image
from displacements import VectorFieldComposer, VectorField, VECTOR_FIELDS

IMAGE_FILES = glob.glob(os.path.join("../tiles", "**/*.png"), recursive=True)

out_per_in = 10
max_images = 100
tile_directory = "../tiles/"
output_directory = "./generated_images/"

os.makedirs(output_directory, exist_ok=True)
random.seed()

num_images = len(IMAGE_FILES)
if num_images > max_images:
    out_per_in = 1
else:
    out_per_in = min(out_per_in, max_images // num_images)

print(f"Generating {out_per_in} images per input image...")

image_idx = 0
for image_path in IMAGE_FILES:
    for _ in range(out_per_in):
        composer = VectorFieldComposer()
        num_fields = random.randint(1, 3)

        available_fields = list(VECTOR_FIELDS.keys())
        for _ in range(num_fields):
            field_type = random.choice(available_fields)
            field = VectorField(name=field_type, field_func=VECTOR_FIELDS[field_type])
            field.randomize()
            composer.fields.append(field)

        base_image = np.array(Image.open(image_path, mode="r"))
        transformed_image = composer.apply_to_image(base_image)

        combined_image = np.hstack(
            [
                np.stack((base_image,) * 3, axis=-1),
                np.stack((transformed_image,) * 3, axis=-1),
                composer.display_fields(base_image.shape[1], base_image.shape[0]),
            ]
        )

        output_path = os.path.join(output_directory, f"gen_{image_idx}.png")
        Image.fromarray(combined_image).save(output_path)

        print(f"Generated image {image_idx + 1}/{num_images}: {output_path}")
        image_idx += 1

    if image_idx >= max_images:
        break
