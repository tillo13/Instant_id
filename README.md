# Batch Image Processing for InstantID Generation

## Overview
The `preprocess.py` module significantly enhances the usability of the `app.py` InstantID generation tool by enabling batch processing of images. With this module, users can now apply various styles and settings to multiple images, streamlining the process of creating stylized, identity-preserving versions of a provided dataset.

## Features
- **Batch Processing:** Process numerous images in one go without the need for repeated manual interventions.
- **Diverse Stylization:** Automatically applies a range of styles to each image, with options for randomness or preset selections.
- **Adjustable Settings:** Each image can be augmented with a different set of parameters, including strength ratios, number of inference steps, and guidance scales.
- **Resize and Padding:** Images are automatically resized and padded to maintain consistent dimensions across the batch.
- **Loop Through Styles:** Optionally configure the script to loop through available styles, ensuring a variety of outputs.
- **Efficient Logging:** Progress and results are logged to a CSV file for easy tracking and auditing.

## Quick Start

1. Place your images in the `sample_images` folder.
2. Configure your desired settings within `preprocess.py`.
3. Run `preprocess.py` to process all images in the folder using the InstantID model.
4. Processed images and logs will be saved to `generated_images` and `generation_log.csv` respectively.

## Configuration Variables

Here's a quick rundown of key configuration variables:

```python
INPUT_FOLDER_NAME = 'sample_images'
OUTPUT_FOLDER_NAME = 'generated_images'
LOG_FILENAME = 'generation_log.csv'
STYLES = [style["name"] for style in style_list]
USE_RANDOM_STYLE = False
NUMBER_OF_LOOPS = 1
...
```

Adjust these settings based on your requirements before executing the script.

## How It Works

`preprocess.py` imports `generate_image` from `app.py` and uses it as the core function to apply InstantID's identity-preserving features to each image in the specified input folder. It handles the batching, resizing, padding, and logging that `app.py` could not originally perform on its own.

### Key Functions
- `resize_and_pad_image`: Standardizes all images to uniform dimensions.
- `log_to_csv`: Records the processing results for each image.
- `initial_image`: Orchestrates the image preprocessing and logging workflow.

### Execution

To start processing, just invoke the `initial_image` function:

```python
from preprocess import initial_image, generate_image
initial_image(generate_image)
```

## Footnotes

### Origin
This module is an addition to the initially single image-based `app.py` script for InstantID: a cutting-edge, zero-shot identity-preserving generation tool developed by the InstantX Team at Xiaohongshu Inc. The tool supports various generative tasks and can customize personal face images with styles and prompts.

### How `app.py` Works
`app.py` allows users to upload a face image, optionally add a pose image, enter a text prompt, apply styles, and generate an identity-preserved output image using the InstantID pipeline. However, the original script lacked batch processing capabilities, which is where `preprocess.py` comes into play.

For more details on `app.py` and InstantID, refer to the official project page and associated research paper:

- Project Page: https://github.com/InstantX/InstantID
- Paper: Wang et al., 2024, "InstantID: Zero-shot Identity-Preserving Generation in Seconds".

---

*InstantID © 2024 by InstantX Team is licensed under Apache License. The research paper was authored by Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu.*