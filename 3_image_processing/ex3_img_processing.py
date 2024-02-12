"""
MyImageProcessor Module

This module defines a class, 'MyImageProcessor', for processing images including color conversion, resizing,
adding frames, finding the center, and detecting faces.

Classes
-------
MyImageProcessor:
    A class for processing images.

Example
-------
image_processor = MyImageProcessor('image.jpg')
image_processor.bgr_2_rgb_convertor()
image_processor.bgr_2_gray_scale_convertor()
image_processor._50_percent_resizer()
image_processor.bgr_image_writer('output_image.jpg')
framed_image = image_processor.frame_it('framed_image.jpg')
center_image = image_processor.find_center('center_image.jpg')
detected_faces_image, num_faces = image_processor.detect_faces()
"""

import cv2
import matplotlib.pyplot as plt


class MyImageProcessor:
    """
    MyImageProcessor Class

    This class provides methods for processing images including color conversion, resizing,
    adding frames, finding the center, and detecting faces.

    Attributes
    ----------
    image_path : str
        Path to the image file.
    image : NumPy.Array
        Loaded image.

    Methods
    -------
    __init__(image_path: str) -> None:
        Constructor for the MyImageProcessor class.
    bgr_2_rgb_convertor() -> np.ndarray:
        Converts the BGR image to RGB and displays it.
    bgr_2_gray_scale_convertor() -> np.ndarray:
        Converts the BGR image to grayscale and displays it.
    _50_percent_resizer() -> np.ndarray:
        Resizes the image to 50% and displays it.
    bgr_image_writer(output_image_path: str) -> None:
        Writes the BGR image to the specified output path.
    frame_it(output_image_with_frame_path: str) -> np.ndarray:
        Adds a red frame around the image and saves it.
    find_center(output_image_with_center: str) -> np.ndarray:
        Finds the center of the image and marks it with a red circle.
    detect_faces() -> tuple:
        Detects faces in the image using a Haar cascade classifier and returns the image with
        detected faces and the number of faces found.

    Example
    -------
    image_processor = MyImageProcessor('image.jpg')
    image_processor.bgr_2_rgb_convertor()
    image_processor.bgr_2_gray_scale_convertor()
    image_processor._50_percent_resizer()
    image_processor.bgr_image_writer('output_image.jpg')
    framed_image = image_processor.frame_it('framed_image.jpg')
    center_image = image_processor.find_center('center_image.jpg')
    detected_faces_image, num_faces = image_processor.detect_faces()
    """
    def __init__(self, image_path):
        """
        Constructor for the MyImageProcessor class.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        None
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)

    def bgr_2_rgb_convertor(self):
        """
        Converts the BGR image to RGB and displays it.

        Returns
        -------
        np.ndarray
            RGB image.
        """
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.show()
        return rgb_image

    def bgr_2_gray_scale_convertor(self):
        """
        Converts the BGR image to grayscale and displays it.

        Returns
        -------
        np.ndarray
            Grayscale image.
        """
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray_image, cmap='gray')
        plt.axis('off')
        plt.show()
        return gray_image

    def _50_percent_resizer(self):
        """
        Resizes the image to 50% and displays it.

        Returns
        -------
        np.ndarray
            Resized RGB image.
        """
        resized_image = cv2.resize(self.image, None, fx=0.5, fy=0.5)
        rgb_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_resized_image)
        plt.axis('off')
        plt.show()
        return rgb_resized_image

    def bgr_image_writer(self, output_image_path):
        """
        Writes the BGR image to the specified output path.

        Parameters
        ----------
        output_image_path : str
            Output path for the BGR image.

        Returns
        -------
        None
        """
        cv2.imwrite(output_image_path, cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

    def frame_it(self, output_image_with_frame_path):
        """
        Adds a red frame around the image and saves it.

        Parameters
        ----------
        output_image_with_frame_path : str
            Output path for the image with the frame.

        Returns
        -------
        np.ndarray
            Image with the added frame.
        """
        framed_image = self.image.copy()
        height, width, _ = framed_image.shape
        cv2.rectangle(framed_image, (0, 0), (width - 1, height - 1), (0, 0, 255), 20)
        cv2.imwrite(output_image_with_frame_path, framed_image)
        return cv2.cvtColor(framed_image, cv2.COLOR_BGR2RGB)

    def find_center(self, output_image_with_center):
        """
        Finds the center of the image and marks it with a red circle.

        Parameters
        ----------
        output_image_with_center : str
            Output path for the image with the center marked.

        Returns
        -------
        np.ndarray
            Image with the marked center.
        """
        center_image = self.image.copy()
        center_x = center_image.shape[1] // 2 # Find center
        center_y = center_image.shape[0] // 2
        cv2.circle(center_image, (center_x, center_y), 5, (255, 0, 0), -1) # Draw + write info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(center_image, 'Image Center', (center_x - 50, center_y + 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(output_image_with_center, cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB))
        return cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

    def detect_faces(self):
        """
        Detects faces in the image using a Haar cascade classifier and returns the image with
        detected faces and the number of faces found.

        Returns
        -------
        tuple
            Tuple containing the image with detected faces and the number of faces found.
        """
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        faces_image = self.image.copy()
        for (x, y, w, h) in faces: # Create rect per each found face
            cv2.rectangle(faces_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        faces_counter = len(faces)
        return cv2.cvtColor(faces_image, cv2.COLOR_BGR2RGB), faces_counter
