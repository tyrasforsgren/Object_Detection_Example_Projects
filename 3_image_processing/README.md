# MyImageProcessor Module

## Overview

The `MyImageProcessor` module provides a set of functionalities for processing images. It includes methods for color conversion, resizing, adding frames, finding the center of the image, and detecting faces using OpenCV.

## Classes

### `MyImageProcessor`

A class for processing images, including color conversion, resizing, adding frames, finding the center, and detecting faces.

#### Methods
- **`__init__(image_path: str) -> None`**: Constructor for the `MyImageProcessor` class. Initializes the image processor with the path to the image file.
- **`bgr_2_rgb_convertor() -> np.ndarray`**: Converts the BGR image to RGB and displays it.
- **`bgr_2_gray_scale_convertor() -> np.ndarray`**: Converts the BGR image to grayscale and displays it.
- **`_50_percent_resizer() -> np.ndarray`**: Resizes the image to 50% of its original size and displays it.
- **`bgr_image_writer(output_image_path: str) -> None`**: Writes the BGR image to the specified output path.
- **`frame_it(output_image_with_frame_path: str) -> np.ndarray`**: Adds a red frame around the image and saves it.
- **`find_center(output_image_with_center: str) -> np.ndarray`**: Finds the center of the image and marks it with a red circle, then saves the image.
- **`detect_faces() -> tuple`**: Detects faces in the image using a Haar cascade classifier and returns the image with detected faces and the number of faces found.

## Example of Use

### Copy the Following Code:

```python
from my_image_processor import MyImageProcessor

# Initialize the image processor with an image file
image_processor = MyImageProcessor('image.jpg')

# Convert the BGR image to RGB and display it
rgb_image = image_processor.bgr_2_rgb_convertor()

# Convert the BGR image to grayscale and display it
gray_image = image_processor.bgr_2_gray_scale_convertor()

# Resize the image to 50% of its original size and display it
resized_image = image_processor._50_percent_resizer()

# Write the BGR image to a file
image_processor.bgr_image_writer('output_image.jpg')

# Add a red frame around the image and save it
framed_image = image_processor.frame_it('framed_image.jpg')

# Find the center of the image, mark it with a red circle, and save it
center_image = image_processor.find_center('center_image.jpg')

# Detect faces in the image, return the image with faces marked and the number of faces detected
detected_faces_image, num_faces = image_processor.detect_faces()
```

#### OR
### Use the DEMO File:
3_image_processing\image_processing_DEMO.ipynb