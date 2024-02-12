"""
Object Detection and Classification Module

This module provides classes for detecting objects in images and videos, classifying them, 
and saving the results in folders.

Classes
-------
ObjectClassification:
    A base class for detecting objects in images and videos.
ImageClassification:
    Subclass of ObjectClassification, specifically for processing images.
VideoClassification:
    Subclass of ObjectClassification, specifically for processing videos.
ImageToVideo_ClassificationAdapter:
    Adapter class to bridge image and video classification.

Constants
---------
LABELS_FILE : str
    Path to the file containing object labels.
WEIGHTS_FILE : str
    Path to the weights file for the YOLOv4 model.
CONFIG_FILE : str
    Path to the configuration file for the YOLOv4 model.
CONFIDENCE_THRESHOLD : float
    Threshold confidence level for object detection.
MIN_DISTANCE : float
    Minimum distance factor for filtering objects.
data_dict : dict
    Dictionary containing categories of objects and their labels.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
LABELS_FILE = "coco.names"
WEIGHTS_FILE = "yolov4.weights"
CONFIG_FILE = "yolov4.cfg"
CONFIDENCE_THRESHOLD = 0.90
MIN_DISTANCE = 1.5

# Classification dictionary
data_dict = {
    'Human': ['person'],
    'Vehicles': ['bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat'],
    'Animal': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    'Sport and Lifestyle': ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
    'Kitchen stuff': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'],
    'Food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
    'In house things': ['chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
}

class ObjectClassification:
    """
    ObjectClassification Class

    A basic class for detecting objects in images and videos.

    Methods
    -------
    get_labels() -> list:
        Returns the list of object labels.
    """
    def __init__(self, input_path): # Attributes are inherited to child classes
        self.input_path = input_path
        self.image = cv2.imread(self.input_path)
        self.labels_file = LABELS_FILE
        self.weights = WEIGHTS_FILE
        self.config = CONFIG_FILE
        self.model = cv2.dnn.readNetFromDarknet(self.config, self.weights)

    def get_labels(self):
        """
        Returns the list of object labels.

        Returns
        -------
        list
            List of object labels.
        """
        return open(self.labels_file).read().strip().split("\n")

class ImageClassification(ObjectClassification):
    """
    ImageClassification Class

    Subclass of ObjectClassification, specifically for processing images.
    """
    def __init__(self, input_path):
        super().__init__(input_path) # Inherits from parent
        self.image_info = [] # Dicts per found obj containing it's details

    def gather_image_info(self):
        """
        Gathers information about objects in the image.

        Returns
        -------
        None
        """
        self.image_info = self.get_image_object_info(self.image)

    def get_image_object_info(self, detection_image):
        """
        Retrieves information about objects detected in the image.

        Parameters
        ----------
        detection_image : np.ndarray
            Image for object detection.

        Returns
        -------
        list
            Information about detected objects.
        """
        # Prep image and send to first layer
        blob = cv2.dnn.blobFromImage(detection_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        layer_names = self.model.getLayerNames()
        # Get output
        output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
        layer_output = self.model.forward(output_layers)
        # Translate output / extract relevant information about found objs
        return self.translate_output(layer_output, detection_image)

    def translate_output(self, layer_output, detection_image):
        """
        Translates the output from the YOLO model to meaningful object information.

        Parameters
        ----------
        layer_output : list
            Output from YOLO model.
        detection_image : np.ndarray
            Image for object detection.

        Returns
        -------
        list
            Information about detected objects.
        """
        height, width, _ = detection_image.shape
        objects_info = []
        layer_names = self.model.getLayerNames()

        # Loop thru output
        for layer_name, output in zip(layer_names, layer_output):
            for detected in output:

                # Extract relevant info:
                scores = detected[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])

                if confidence > CONFIDENCE_THRESHOLD: # Filter by confidence
                    box = detected[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, box_width, box_height) = box.astype("int")
                    x = int((center_x - box_width / 2))
                    y = int((center_y - box_height / 2))

                    category_name = None # Extratc category name
                    for category, classes in data_dict.items():
                        if self.get_labels()[class_id] in classes:
                            category_name = category
                            break

                    obj_info = { # Create dict of details
                        'class_id': class_id,
                        'category': category_name,
                        'layer_name': self.get_labels()[class_id],
                        'confidence': confidence,
                        'box': (x, y, box_width, box_height)
                    }

                    objects_info.append(obj_info)
        # Filter overlapping boxes :
        objects_info = self.filter_boxes_by_distance(objects_info)
        return objects_info

    def filter_boxes_by_distance(self, objects_info):
        """
        Filters objects based on their distance from each other.

        Parameters
        ----------
        objects_info : list
            Information about detected objects.

        Returns
        -------
        list
            Filtered list of detected objects.
        """
        # Prioirtize most conficence objs by sorting them
        sorted_boxes = sorted(objects_info, key=lambda x: x['confidence'], reverse=True)
        filtered_boxes = []

        for box in sorted_boxes:
            x, y, width, height = box['box'] # Get the box
            is_close = False

            for filtered_box in filtered_boxes:
                filtered_x, filtered_y, filtered_width, filtered_height = filtered_box['box']
                
                # Calculate distance between box centers
                center_distance = ((x + width / 2) - (filtered_x + filtered_width / 2)) ** 2 + ((y + height / 2) - (filtered_y + filtered_height / 2)) ** 2
                min_distance = min(width, filtered_width) * MIN_DISTANCE 
                
                if center_distance < min_distance:
                    is_close = True
                    break

            if not is_close:
                filtered_boxes.append(box)

        return filtered_boxes

    def draw_boxes_on_image(self):
        """
        Draws bounding boxes and labels on the image.

        Returns
        -------
        np.ndarray
            Image with bounding boxes and labels.
        """
        box_image = self.image.copy()
        for obj_info in self.image_info:
            # Get the details from the obj database
            class_id = obj_info['class_id']
            confidence = obj_info['confidence']
            x, y, box_width, box_height = obj_info['box']

            # Color font ect details
            color = (255, 255, 0)
            thickness = 2
            font_scale = 1
            label = self.get_labels()[class_id]

            # Draw text and rectangle/box
            cv2.rectangle(box_image, (x, y), (x + box_width, y + box_height), color, thickness)
            cv2.putText(box_image, f"{label}: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return box_image

    def save_boxes_to_folders(self): 
        """
        Saves detected objects to folders based on their categories.

        Returns
        -------
        None
        """
        for category in data_dict.keys(): # Save boxed image at respective category folder
            folder_path = f'classification_folders/{category}'
            os.makedirs(folder_path, exist_ok=True)

    def display_image(self, image):
        """
        Displays the image.

        Parameters
        ----------
        image : np.ndarray
            Image.

        Returns
        -------
        None
        """
        if isinstance(image, np.ndarray): # Error check
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print("Error: Invalid image format. Cannot display.")

class VideoClassification(ObjectClassification):
    """
    VideoClassification Class

    Subclass of ObjectClassification, specifically for processing videos.
    """
    def __init__(self, input_path, image_classifier):
        super().__init__(input_path) # Inherits parent attribs
        self.image_classifier = image_classifier

    def box_video_frames(self, frames):
        """
        Boxes each frame in the video with detected objects.

        Parameters
        ----------
        frames : list
            List of video frames.

        Returns
        -------
        list
            List of video frames with bounding boxes and labels.
        """
        boxed_frames = []
        for frame in frames: # For each frame get all objs and box them
            self.image_classifier.image = frame
            self.image_classifier.gather_image_info()
            boxed_frame = self.image_classifier.draw_boxes_on_image()
            boxed_frames.append(boxed_frame)
            # Create a list containing all frames and all thier object infos
        
        return boxed_frames

    def display_video_frames(self, frames, video_name="output_video.mp4", frame_rate=30, save_video=False):
            """
            Displays the video frames with detected objects and optionally saves the annotated video.

            Parameters
            ----------
            frames : list
                List of video frames with bounding boxes and labels.
            output_file : str, optional
                Output filename for the annotated video, by default "output_video.mp4".
            frame_rate : int, optional
                Frame rate of the output video, by default 30.
            save_video : bool, optional
                Whether to save the annotated video, by default False.

            Returns
            -------
            None
            """

            output_file = video_name
            if not frames: # Error detection
                print("Error: Frames list is empty.")
                return
            
            height, width, _ = frames[0].shape # Create a video with size of the frames
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

            for frame in frames:
                if frame is not None:
                    out.write(frame)

            out.release()

            if save_video:
                video_capture = cv2.VideoCapture(output_file) # Start cap
                while True:  
                    ret, frame = video_capture.read()
                    if not ret:
                        video_capture.release()
                        video_capture = cv2.VideoCapture(output_file)
                        continue
                    cv2.putText(frame, "Press 'ESC' to close", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    cv2.imshow('Video', frame) # Check to end it
                    if cv2.waitKey(25) & 0xFF == 27:
                        break

                video_capture.release() # End video n close window
                cv2.destroyAllWindows()
            else:
                print("Video display complete.")

class ImageToVideo_ClassificationAdapter: 
    """
    ImageToVideo_ClassificationAdapter Class

    Adapter class to convert images to videos and classify objects.
    """
    def __init__(self, input_path, image_classifier):
        self.input_path = input_path
        self.image_classifier = image_classifier
        self.all_frame_obj_info = {}

    def gather_video_info(self, frames):
        """
        Gathers information about objects in the video frames.

        Parameters
        ----------
        frames : list
            List of video frames.

        Returns
        -------
        None
        """
        self.detect_objects(frames)

    def get_all_frames_from_vid(self):
        """
        Retrieves all frames from the video.

        Returns
        -------
        list
            List of video frames.
        """
        video_capture = cv2.VideoCapture(self.input_path)
        frames = []

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frames.append(frame)

        video_capture.release()
        return frames

    def detect_objects(self, frames):
        """
        Detects objects in the video frames.

        Parameters
        ----------
        frames : list
            List of video frames.

        Returns
        -------
        None
        """
        for i, frame in enumerate(frames):
            # as there is a wait, indicate the program is working
            print(f'Processing {i+1}/{len(frames)} frames.')
            if frame is None or frame.size == 0:
                print("Empty frame detected.")
                continue
            # Do obj detection for all frames and save the data
            self.image_classifier.image = frame
            self.image_classifier.gather_image_info()
            self.all_frame_obj_info[f'Frame {i}'] = self.image_classifier.image_info

