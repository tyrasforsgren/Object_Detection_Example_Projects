# Object Detection and Classification Module

## Overview

The `Object Detection and Classification Module` provides a comprehensive solution for detecting and classifying objects in images and videos. It supports object detection using the YOLOv4 model, and includes functionality for processing images and videos, and saving results to specific folders based on object categories.

## Classes

### `ObjectClassification`

A base class for detecting objects in images and videos.

#### Methods
- **`get_labels() -> list`**: Returns the list of object labels.

### `ImageClassification`

Subclass of `ObjectClassification`, specifically for processing images.

#### Methods
- **`gather_image_info() -> None`**: Gathers information about objects in the image.
- **`get_image_object_info(detection_image: np.ndarray) -> list`**: Retrieves information about objects detected in the image.
- **`translate_output(layer_output: list, detection_image: np.ndarray) -> list`**: Translates YOLO model output to meaningful object information.
- **`filter_boxes_by_distance(objects_info: list) -> list`**: Filters objects based on their distance from each other.
- **`draw_boxes_on_image() -> np.ndarray`**: Draws bounding boxes and labels on the image.
- **`save_boxes_to_folders() -> None`**: Saves detected objects to folders based on their categories.
- **`display_image(image: np.ndarray) -> None`**: Displays the image.

### `VideoClassification`

Subclass of `ObjectClassification`, specifically for processing videos.

#### Methods
- **`box_video_frames(frames: list) -> list`**: Boxes each frame in the video with detected objects.
- **`display_video_frames(frames: list, video_name="output_video.mp4", frame_rate=30, save_video=False) -> None`**: Displays the video frames with detected objects and optionally saves the annotated video.

### `ImageToVideo_ClassificationAdapter`

Adapter class to convert images to videos and classify objects.

#### Methods
- **`gather_video_info(frames: list) -> None`**: Gathers information about objects in the video frames.
- **`get_all_frames_from_vid() -> list`**: Retrieves all frames from the video.
- **`detect_objects(frames: list) -> None`**: Detects objects in the video frames.

## Constants

- **`LABELS_FILE`**: Path to the file containing object labels.
- **`WEIGHTS_FILE`**: Path to the weights file for the YOLOv4 model.
- **`CONFIG_FILE`**: Path to the configuration file for the YOLOv4 model.
- **`CONFIDENCE_THRESHOLD`**: Threshold confidence level for object detection.
- **`MIN_DISTANCE`**: Minimum distance factor for filtering objects.
- **`data_dict`**: Dictionary containing categories of objects and their labels.

## Example of Use

### Image Classification

#### Copy the Following Code:

```python
from object_detection import ImageClassification

# Initialize image classifier
image_classifier = ImageClassification('image.jpg')

# Gather information about objects in the image
image_classifier.gather_image_info()

# Draw bounding boxes and labels on the image
boxed_image = image_classifier.draw_boxes_on_image()

# Save detected objects to folders
image_classifier.save_boxes_to_folders()

# Display the image with bounding boxes
image_classifier.display_image(boxed_image)
```
##### OR
#### Use the DEMO File:
2_object_detection_and_video\adapter_DEMO.ipynb