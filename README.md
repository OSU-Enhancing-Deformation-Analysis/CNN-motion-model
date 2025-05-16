# Motion Vector Prediction Model

> This README specifically details the `batch-m4-combo.py` file.
[Combo Model](https://github.com/OSU-Enhancing-Deformation-Analysis/CNN-motion-model/blob/main/batch_models/batch-m4-combo.py)

## Overview

This model predicts 2D motion vector fields between two input images. The primary goal is to learn the transformation that maps an initial image to a subsequent image. This is achieved by training a U-Net-like convolutional neural network with self-attention mechanisms. A key feature of this project is its sophisticated synthetic data generation pipeline, which generates diverse training samples by programmatically creating and combining vector fields, and using various geometric and procedural shapes to mask these fields.

The model is trained using PyTorch, and experiment tracking is managed with Weights & Biases (Wandb).

## Features

*   **Vector Fields:** Generates synthetic motion vector fields using a composable system of base vector fields (translation, rotation, scale, shear, shear2, gradient, gradient2, harmonic, harmonic2, pearling, vortex, vortex2, perlin/simplex noise, swirl).
*   **Shape Masks:** Adds generated 2D shapes (squares, circles, blobs, swirls, polygons, Perlin noise, Voronoi patterns, etc.) that are warped and used to create localized variations and "sharpness" in the target vector fields.
*   **U-Net Architecture with Self-Attention:** Creates a U-Net like encoder-decoder structure, with added self-attention layers in the bottleneck.
*   **End-to-End Training:** Takes two images (original and target/warped) as input and directly regresses the 2D vector field.
*   **Weights & Biases Integration:** Logs metrics, configuration, model checkpoints, and visual samples during training.
*   **Customizable Training:** Supports training for a fixed number of epochs or a maximum duration.

## Standard Project Information

### Software

*   Python 3.8+
*   `pip` (Python package installer)

### Hardware

*   A CUDA-enabled GPU is highly recommended for reasonable training times. The script automatically detects and uses an available GPU.
*   The `BATCH_SIZE` is dynamically calculated based on available GPU memory, but a GPU with significant VRAM (e.g., >16GB) is beneficial for larger batch sizes.

### Python Packages

Install the required packages using pip:

```bash
pip install numpy matplotlib perlin-numpy Pillow wandb torch scipy
```

### 1. Directory Structure

Create the following directory structure:

```
.
├── your_script_name.py  # This script
├── tiles/                 # Directory for input image tiles
│   ├── sequence_name1/    # e.g., "g60"
│   │   ├── image_folder1/   # e.g., "0001.tif" (original image name, now a folder)
│   │   │   ├── tile_0.png
│   │   │   ├── tile_1.png
│   │   │   └── ...
│   │   ├── image_folder2/
│   │   │   └── ...
│   │   └── ...
│   └── sequence_name2/
│       └── ...
└── <MODEL_NAME>/          # Directory created by the script to save model files (e.g., b4-unknown-test/)
    └── <MODEL_NAME>.pth   # Saved model checkpoint
```

*   `your_script_name.py`: The Python script provided.
*   `tiles/`: This is the root directory for your input image data. The script expects images to be organized in subdirectories.
*   `<MODEL_NAME>/`: A directory will be automatically created based on the `MODEL_NAME` (either a default or passed as a command-line argument) to store the trained model weights (`.pth` file).

### 2. Weights & Biases (Wandb) Account

This script uses Weights & Biases for experiment tracking, visualization, and model artifact storage. You can remove this part if you don't want this kind of logging.

1.  **Create an Account:** If you don't have one, sign up for a free account at [wandb.ai](https://wandb.ai).
2.  **Login:** Log in to your Wandb account from your terminal:
    ```bash
    wandb login
    ```
    You'll be prompted for your API key.

The script will automatically create a new project named "motion-model" (or use an existing one) in your Wandb account and log runs under the specified `MODEL_NAME`.

### 3. Input Tiles Data (`./tiles` directory)

The model trains on pairs of images: an "original" image and a "warped" version of it, alongside the "ground truth" vector field that describes the warp. The `CustomDataset` class in this script *synthetically generates* the warped images and vector fields. The images in the `./tiles` directory serve as the **base images** for this synthetic generation.

*   **Format:** PNG images are expected. The script loads them as grayscale.
*   **Size:** The `TILE_SIZE` constant (default 256) determines the expected dimensions (e.g., 256x256 pixels).
*   **Content:** These should be diverse textures or patterns that the model can learn.
*   **Organization:**
    *   The script uses `glob.glob(os.path.join(TILES_DIR, "**/*.png"), recursive=True)` to find all PNG files within the `TILES_DIR`.
    *   A more specific loading mechanism (`sequence_arrays`) is also implemented, expecting a structure like `TILES_DIR/<sequence_name>/<image_folder_name>/tile_<number>.png`. This is used for generating specific test samples during validation.

**Generating Tiles (External Process):**

This script *consumes* tiles; it does not generate the initial set of `.png` files from larger images. To split up larger images into the appropriate tile sizes with some overlap, there are some other file around here somewhere that do that but need to be modified to actually have the images. [Code from the Model Output Display](https://github.com/OSU-Enhancing-Deformation-Analysis/Model-Output-Preview/blob/main/machine_learning_model.py#L17-L98)

## Configuration

Key configuration options are defined as constants at the beginning of the script.

| Parameter              | Location in Code | Description                                                                                                                               | Default Value                 |
| :--------------------- | :------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------- |
| `device`               | 39                              | Auto-detects CUDA GPU or defaults to CPU. You might need to set this manually to "cuda" or "mps" depending on the device.                 | auto                          |
| `TILES_DIR`            | 48                              | Path to the directory containing input image tiles.                                                                                       | `"./tiles"`                   |
| `MAX_TILES`            | 52                              | Maximum number of base tile images to load from `TILES_DIR`. Set lower for quick tests.                                                   | `17556`                       |
| `TILE_SIZE`            | 55                              | Dimension of the square input tiles (e.g., 256 for 256x256).                                                                              | `256`                         |
| `EPOCHS`               | 64                              | Number of training epochs. If `MAX_TIME` is set, this won't be used.                                                                      | `None` (or e.g., `10`)        |
| `MAX_TIME`             | 66                              | Maximum training time in seconds. If set, training stops when this duration is exceeded, regardless of `EPOCHS`.                          | `100 * 60 * 60` (100 hours)   |
| `BATCH_SIZE`           | 69                              | Number of samples per training batch. Dynamically calculated based on GPU memory. Formula: `int((GPU_MEMORY - 1.5) / 0.13 / 6)`.          | dynamic                       |
| `LEARNING_RATE`        | 72                              | Initial learning rate for the Adam optimizer.                                                                                             | `0.0001`                      |
| `SAVE_FREQUENCY`       | 73                              | Save a model checkpoint and log Wandb samples every N epochs. Best to keep this at 1 incase something happens during traning.             | `1`                           |
| `MODEL_NAME`           | 76                              | Name for the model, used for saving files and in Wandb. Can be set via command-line argument.                                             | `"b4-unknown-test"` or `argv[1]`|
| `NUM_WORKERS`          | 1154                            | Number of worker processes for data loading.                                                                                              | `4`                           |

**Note on `BATCH_SIZE` Calculation:**
The formula `int((GPU_MEMORY - 1.5) / 0.13 / 6)` is tuned for this specific model architecture and `TILE_SIZE` using trial and error and Desmos.
*   `GPU_MEMORY`: Total GPU memory in GB.
*   `- 1.5`: A buffer subtracted from total memory (e.g., for OS, display, other processes).
*   `/ 0.13`: Division to bring it down based on Desmos curve fit.
*   `/ 6`: Manual adjustment because it was off.

## Code Architecture

### 1. Data Generation

The core of the data generation lies in the `CustomDataset` class, which synthesizes training samples on-the-fly.

#### Vector Fields (`vector_field` decorator, `VectorField` class, `VectorFieldComposer` class)

*   **Purpose:** To programmatically generate diverse 2D vector fields (dU, dV components) which represent motion or deformation.
*   **How it works:**
    *   The `@vector_field()` decorator registers various functions (e.g., `translation_field`, `rotation_field`, `perlin_field`, `swirl_field`) into a global `VECTOR_FIELDS` dictionary. Each function defines a basic type of 2D field.
    *   The `VectorField` dataclass encapsulates a single field instance, allowing randomization of its `amplitude`, `center`, `scale`, and `rotation`. Its `apply` method computes the (dx, dy) components for a given grid.
    *   `VectorFieldComposer` manages a list of `VectorField` instances. It can `add_field`, `pop_field`, and `compute_combined_field` by summing the contributions of all active fields. The `apply_to_image` method uses the combined field to warp an input image using `scipy.ndimage.map_coordinates`.
*   **Justification:** This system provides a flexible and extensible way to create an almost infinite variety of complex vector fields for data augmentation, which is crucial for training a robust model that can generalize to unseen motion patterns.
*   **Quirks:**
    *   The displacement in `apply_to_image` (`new_x = self.pos_x - dU * 10`) has a scaling factor of `10`. This factor controls the magnitude of the warp; a larger value means more extreme deformations.

#### Shape Generation Functions (`create_square_shape`, `create_circle_shape`, etc.)

*   **Purpose:** To generate a wide variety of 2D grayscale shape masks (e.g., squares, circles, blobs, swirls, gradients, checkers, polygons, Perlin noise, Voronoi patterns, stripes).
*   **How it works:** Each `create_<shape_name>_shape` function generates a NumPy array of `TILE_SIZE` x `TILE_SIZE`. These functions typically randomize parameters like position, scale, rotation, and other shape-specific attributes (e.g., number of sides for polygons, checker size).
*   **Justification:** These shapes are used to modulate the generated vector fields (see `CustomDataset`), introducing localized variations, sharp transitions, and structured patterns into the motion. This helps the model learn to predict more complex and less uniform fields.
*   **Quirks:**
    *   Many functions use `scipy.ndimage.rotate` for rotation, which can be computationally intensive.
    *   The grayscale values assigned within the shapes (e.g., 200, 150) are somewhat arbitrary but serve to define the mask's intensity.

#### `CustomDataset(Dataset)` Class

*   **Purpose:** This PyTorch `Dataset` generates training/validation samples dynamically. Each sample consists of an (original_image, warped_image) pair and the corresponding ground_truth_vector_field.
*   **How it works (`__getitem__` method):**
    1.  **Image Selection:** An image is loaded from `TILE_IMAGE_PATHS`.
    2.  **Base Vector Field:** A base vector field is created by randomly choosing 1 or 2 field types from `VECTOR_FIELDS` (e.g., perlin, rotation) and composing them using `VectorFieldComposer`. Their parameters (amplitude, scale, etc.) are randomized.
    3.  **Shape Masking (Probabilistic):** With a 75% chance, a shape-based modulation is applied:
        *   A random shape function (e.g., `create_circle_shape`, `create_perlin_noise_shape`) is chosen to generate a `shape_layer` (a grayscale mask).
        *   This `shape_layer` itself is warped using another randomly generated vector field (`shape_morph_composer`). This creates a `morphed_shape`.
        *   Optionally (30% chance), the `morphed_shape` is blurred using `gaussian_filter` and normalized.
        *   The `morphed_shape` (normalized to 0-1) is then used to combine vector fields:
            *   **50% chance (Invert):** The final field is `(1 - morphed_shape) * (base_field * -1) + morphed_shape * base_field`. This effectively inverts the base field outside the shape and keeps it normal inside.
            *   **50% chance (Replace):** The final field is `(1 - morphed_shape) * base_field + morphed_shape * another_random_field`. This keeps the base field outside the shape and applies a *different* random field inside the shape.
    4.  **No Shape Masking:** With a 25% chance, the `final_field` is simply the initially computed `base_vector_field`.
    5.  **Image Warping:** The selected original image is warped using the `final_field` (dU, dV components) via `scipy.ndimage.map_coordinates` to produce the `warped_image`. The `order=0` (nearest-neighbor interpolation) and `mode="wrap"` are used.
    6.  **Output:** Returns `(original_image, warped_image)` as `X` and `(dU, dV)` of the `final_field` as `y`.
*   **Justification:** This sophisticated pipeline aims to generate highly diverse training data. The key idea of using warped shapes to modulate vector fields is designed to introduce "sharpness" and complex local variations, pushing the model to learn finer details in motion.
*   **Quirks:**
    *   **Random Seeding:** `random.seed(index + (current_epoch * NUM_TILES * self.variations_per_image))` ensures that for a given base image, different variations are generated across different epochs during training. For validation, a fixed offset is added to the seed for reproducibility.
    *   **Vector Field Scaling:** The `dU, dV` components are scaled by `* 10` before being used in `map_coordinates`. This controls the magnitude of the pixel displacements. Without it the nearest-neighbor interpolation of the image warping doesn't move a large number of the pixels because the magnitude of displacement is too small.

### 2. Model (`ComboMotionVectorRegressionNetwork`)

The neural network is a U-Net like architecture designed for image-to-image regression tasks.

#### `ConvolutionBlock(nn.Module)`

*   **Purpose:** A standard reusable convolutional block.
*   **How it works:** Consists of a 2D Convolutional layer, Batch Normalization, and LeakyReLU activation. It includes a residual (skip) connection. If input and output channels differ, the residual path uses a 1x1 convolution and Batch Normalization to match dimensions.
*   **Justification:** Modular design, promotes stable training via batch norm, and residual connections help with gradient flow in deeper networks.

#### `SelfAttention(nn.Module)`

*   **Purpose:** Implements a self-attention mechanism (similar to non-local blocks) to allow the model to capture long-range dependencies in the feature maps.
*   **How it works:**
    1.  Input `x` is projected into `query`, `key`, and `value` representations using 1x1 convolutions. Query and key channels are reduced by a factor of 8.
    2.  Attention scores are computed by `softmax(query @ key.T)`.
    3.  The attention scores are used to weight the `value` projection: `attention_scores @ value`.
    4.  The result is passed through another 1x1 convolution (`o_conv`).
    5.  A learnable `gamma` parameter scales the attention output, which is then added back to the original input `x` (residual connection).
*   **Justification:** Enables the model to weigh information from different spatial locations, improving its ability to understand global motion and complex motion patterns that span large areas of the image.

#### `ComboMotionVectorRegressionNetwork(nn.Module)`

*   **Purpose:** The main network that takes two input images (stacked along the channel dimension, total `input_images=2` channels) and predicts a 2-channel output representing the (dU, dV) motion vector field.
*   **Architecture:**
    *   **Encoder (Downsampling Path):**
        *   `conv1`, `pool1`: Initial feature extraction.
        *   `conv2`, `pool2`: Further feature extraction and downsampling.
        *   `conv_layers_down`: A sequence of `ConvolutionBlock`s and `MaxPool2d` layers to create a deeper encoder, progressively reducing spatial resolution and increasing channel depth. Feature maps from intermediate `ConvolutionBlock`s are stored for skip connections (`down_features_to_concat`, `intermediate_features`).
    *   **Bottleneck:**
        *   `attention`: A `SelfAttention` block is applied to the most compressed feature representation from the encoder.
    *   **Decoder (Upsampling Path):**
        *   Symmetrically mirrors the encoder. Uses `ConvTranspose2d` for upsampling.
        *   After each upsampling step, the feature map is concatenated with the corresponding feature map from the encoder path (skip connection).
        *   `ConvolutionBlock`s (`conv_up1` to `conv_up4`) are used to refine the upsampled and concatenated features.
    *   **Output Layer:**
        *   `output_conv`: A final 3x3 convolution that maps the refined features from the last decoder stage to 2 output channels (for dU and dV).
*   **Justification:** U-Nets are well-suited for pixel-wise regression tasks like motion estimation because skip connections allow the decoder to leverage low-level features from the encoder, preserving spatial detail, while the bottleneck learns correlation information. The "Combo" name was made up because it was a slight combination of two other models I tried.
*   **Quirks:** The precise number of blocks, channel sizes, and the exact points of skip connections are specific design choices. Careful indexing is required to ensure correct feature maps are concatenated (e.g., `down_features_to_concat[-2]`). It is not very flexible to changes in size without some adjustments.

### 3. Training Process

#### Loss Function (`custom_loss`)

*   **Purpose:** To measure the discrepancy between the predicted vector field and the ground truth vector field.
*   **How it works:** It calculates the mean End-Point Error (EPE).
    1.  `squared_difference = (predicted_vectors - target_vectors)**2`
    2.  `epe_map = torch.sqrt(squared_difference.sum(dim=1))` calculates the L2 norm (Euclidean distance) of the error vector `(dU_err, dV_err)` for each pixel. This results in a map of EPE values.
    3.  `epe_loss = epe_map.mean()` computes the average EPE across all pixels in the batch.
*   **Justification:** EPE is a standard metric for optical flow and motion vector estimation, directly representing the average pixel displacement error.

#### Optimizer and Scheduler

*   **Optimizer:** `torch.optim.Adam` with `LEARNING_RATE` is used for updating model weights.
*   **Scheduler:** `torch.optim.lr_scheduler.ReduceLROnPlateau` is used to decrease the learning rate by a factor of 0.1 if the validation loss (`average_validation_loss`) does not improve for 5 epochs (`patience=5`). This helps in fine-tuning the model towards the end of training.

#### Weights & Biases (Wandb) Logging

*   **Initialization:** `wandb.init()` sets up the experiment run, logging hyperparameters defined in `wandb_config`.
*   **Model Watching:** `wandb.watch(model)` tracks gradients and parameters of the model.
*   **Metrics Logging:**
    *   **Batch-level:** `train_loss`, `gradient_norm`, `learning_rate`.
    *   **Epoch-level:** `training_loss`, `validation_loss`, `learning_rate`.
*   **Sample Visualization:**
    *   At each `SAVE_FREQUENCY`, a fixed set of validation samples (`samples_images`, `samples_vectors`) are processed by the current model.
    *   The original image, the ground truth warped image, the ground truth vector field (visualized as an RGB image), and the predicted vector field (visualized as an RGB image) are combined and logged to Wandb.
    *   Additionally, "test" samples are generated using specific sequences from `sequence_arrays` (e.g., "g69" sequence) to visualize performance on actual sequential image pairs.
*   **Alerts:** Wandb alerts are sent at the start and end of training.
*   **Artifacts:** The final trained model (`MODEL_FILE`) is saved as a Wandb artifact.

#### Training Loop

*   The loop continues until `EPOCHS` are completed or `MAX_TIME` is exceeded.
*   **Training Phase (`model.train()`):**
    *   Iterates through `training_dataloader`.
    *   For each batch:
        1.  Moves data to `device`.
        2.  Performs a forward pass: `pred = model(batch_images)`.
        3.  Calculates loss: `loss = custom_loss(pred, batch_vectors)`.
        4.  Performs backpropagation: `loss.backward()`.
        5.  Clips gradients: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to prevent exploding gradients.
        6.  Updates model weights: `optimizer.step()`.
        7.  Resets gradients: `optimizer.zero_grad()`.
*   **Validation Phase (`model.eval()`):**
    *   Iterates through `validation_dataloader` with `torch.no_grad()` to disable gradient calculations.
    *   Computes `validation_loss` for each batch.
*   **Checkpointing:** `torch.save(model.state_dict(), "snapshot_save.pt")` saves a temporary snapshot. The final model is saved to `MODEL_FILE`.

## Running the Script

1.  **Ensure Prerequisites:** Verify Python environment, installed packages, GPU drivers (if using GPU), and Wandb login.
2.  **Prepare Data:** Place your input tile images in the `./tiles` directory as described under "Input Tiles Data".
3.  **Execute from Terminal:**
    ```bash
    python your_script_name.py [model_run_name]
    ```
    *   `your_script_name.py`: The name of this Python script.
    *   `[model_run_name]` (optional): A name for this training run. If provided, it will be used as `MODEL_NAME` for saving files and for the Wandb run name. If not provided, a default like "b4-unknown-test" will be used.

    Example:
    ```bash
    python train_motion_model.py microfluidics_run_01
    ```

## Outputs

*   **Trained Model:**
    *   A model checkpoint file named `<MODEL_NAME>.pth` will be saved in the `<MODEL_NAME>/` directory (e.g., `b4-unknown-test/b4-unknown-test.pth`).
    *   A temporary snapshot `snapshot_save.pt` is saved during training at `SAVE_FREQUENCY`.
    *   The final model is also logged as an artifact to Weights & Biases.
*   **Weights & Biases Dashboard:**
    *   A link to the Wandb run will be printed in the console when training starts.
    *   This dashboard will show:
        *   Logged metrics (training/validation loss, learning rate, gradient norm).
        *   System metrics (GPU/CPU utilization).
        *   Model configuration.
        *   Visualizations of sample predictions.
        *   Saved model artifacts.

## Potential Improvements / Future Work

*   **Hyperparameter Tuning:** Systematically tune learning rate, optimizer parameters, and network architecture details.
*   **Data Normalization:** Explicitly normalize input images to the model (e.g., to [0, 1] or standardize).
*   **Advanced Vector Field Visualization:** Use HSV color mapping for more intuitive visualization of predicted vector fields in Wandb.
*   **More Sophisticated Augmentations:** Explore photometric augmentations (brightness, contrast) on the input images.
*   **Loss Function Exploration:** Experiment with other loss functions, such as those robust to outliers or those that consider spatial smoothness of the vector field.
*   **Transfer Learning:** If applicable, pre-train parts of the network on larger image datasets.
*   **Evaluate on Real-World Data:** Rigorously test the model on benchmark datasets for optical flow or motion estimation if available and relevant to the application domain.
