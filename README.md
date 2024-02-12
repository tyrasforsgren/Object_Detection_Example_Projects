# Object Detection and Classification Module

This module provides classes for detecting objects in images and videos, classifying them, 
and saving the results in folders.

## Classes

- **ObjectClassification**: A base class for detecting objects in images and videos.
- **ImageClassification**: Subclass of ObjectClassification, specifically for processing images.
- **VideoClassification**: Subclass of ObjectClassification, specifically for processing videos.
- **ImageToVideo_ClassificationAdapter**: Adapter class to bridge image and video classification.

## Constants

- **LABELS_FILE**: Path to the file containing object labels.
- **WEIGHTS_FILE**: Path to the weights file for the YOLOv4 model.
- **CONFIG_FILE**: Path to the configuration file for the YOLOv4 model.
- **CONFIDENCE_THRESHOLD**: Threshold confidence level for object detection.
- **MIN_DISTANCE**: Minimum distance factor for filtering objects.
- **data_dict**: Dictionary containing categories of objects and their labels.

## Usage Example

```python
import cv2

# Initialize an ImageClassification object
image_classifier = ImageClassification("input_image.jpg")

# Gather information about objects in the image
image_classifier.gather_image_info()

# Draw bounding boxes and labels on the image
boxed_image = image_classifier.draw_boxes_on_image()

# Save detected objects to folders based on their categories
image_classifier.save_boxes_to_folders()

# Display the image with bounding boxes and labels
image_classifier.display_image(boxed_image)
```


# RegularShapeDetector Module

This module defines a class, 'RegularShapeDetector', for detecting and analyzing regular shapes from an image. It uses contour analysis and has methods for finding shapes, saving results, and displaying the original image.

## Classes

- **RegularShapeDetector**: A class for detecting regular shapes in an image.

## Example of Use

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Initialize a RegularShapeDetector object
shape_detector = RegularShapeDetector('shapes.jpg')

# Detect regular shapes in the image
shape_detector.find_shapes()

# Save the detected shape data to a CSV file
shape_detector.save_results_as_df('shape_data.csv')

# Display the original image
shape_detector.display_image()
```

# MyImageProcessor Module

This module defines a class, 'MyImageProcessor', for processing images including color conversion, resizing,
adding frames, finding the center, and detecting faces.

## Classes

- **MyImageProcessor**: A class for processing images.

## Example of Use

```python
import cv2
import matplotlib.pyplot as plt

# Initialize a MyImageProcessor object
image_processor = MyImageProcessor('image.jpg')

# Convert the BGR image to RGB and display it
image_processor.bgr_2_rgb_convertor()

# Convert the BGR image to grayscale and display it
image_processor.bgr_2_gray_scale_convertor()

# Resize the image to 50% and display it
image_processor._50_percent_resizer()

# Write the BGR image to the specified output path
image_processor.bgr_image_writer('output_image.jpg')

# Add a red frame around the image and save it
framed_image = image_processor.frame_it('framed_image.jpg')

# Find the center of the image and mark it with a red circle
center_image = image_processor.find_center('center_image.jpg')

# Detect faces in the image using a Haar cascade classifier
detected_faces_image, num_faces = image_processor.detect_faces()
