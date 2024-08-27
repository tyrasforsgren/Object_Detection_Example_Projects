# RegularShapeDetector Module

## Overview

The `RegularShapeDetector` module is designed for detecting and analyzing regular shapes within an image using contour analysis techniques. It provides methods for identifying shapes, saving results, and displaying the original image. This module can be particularly useful for computer vision applications requiring shape detection and analysis.

## Key Features

- **Shape Detection**: Identifies regular shapes such as circles, squares, triangles, and more from an image.
- **Shape Analysis**: Determines the regularity of shapes based on corner and side analysis.
- **Result Export**: Saves the detected shape information into a CSV file.
- **Image Display**: Visualizes the original image for easy verification of detected shapes.

## Classes

### `RegularShapeDetector`

A class for detecting regular shapes in an image using contour analysis.

#### Attributes:
- `image` (NumPy Array): The image containing shapes.
- `image_path` (str): Path to the image file.

## Methods

1. **`__init__(image_path: str) -> None`**: Initializes the `RegularShapeDetector` class with the path to the image file.

2. **`find_shapes() -> dict`**: Detects regular shapes in the image using contour analysis and returns a dictionary containing shape information.

3. **`determine_shape(num_corners: int) -> str`**: Determines the shape name based on the number of corners detected.

4. **`is_regular_shape(approx: np.ndarray) -> bool`**: Checks if the shape represented by the given corners is regular (all sides equal).

5. **`circular_specification(approx: np.ndarray) -> tuple`**: Determines if a shape is a circle or an oval based on aspect ratio analysis.

6. **`is_square(approx: np.ndarray) -> bool`**: Checks if the shape represented by the given corners is a square (aspect ratio close to 1).

7. **`save_results_as_df(path_to_save: str) -> None`**: Saves the detected shape data to a CSV file.

8. **`display_image() -> None`**: Displays the original image using Matplotlib.

## Example of Use

### Copy the Following Code:

```python
# Import the module
from regular_shape_detector import RegularShapeDetector

# Initialize the shape detector with the path to an image
shape_detector = RegularShapeDetector('shapes.jpg')

# Detect shapes in the image
shape_detector.find_shapes()

# Save the detected shapes to a CSV file
shape_detector.save_results_as_df('shape_data.csv')

# Display the original image
shape_detector.display_image()
```

#### OR

### Use the DEMO File:
1_shape_detection\shape_detector_DEMO.ipynb
