"""
RegularShapeDetector Module

This module defines a class, 'RegularShapeDetector', for detecting and analyzing
regular shapes from an image. It uses contour analysis and has methods for finding
shapes, saving results, and displaying the original image.

Classes
-------
RegularShapeDetector:
    A class for detecting regular shapes in an image.

Example of use
--------------
shape_detector = RegularShapeDetector('shapes.jpg')
shape_detector.find_shapes()
shape_detector.save_results_as_df('shape_data.csv')
shape_detector.display_image()

Global functions
----------------
None
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

class RegularShapeDetector:
    """
    RegularShapeDetector Class

    This class provides methods for detecting and analyzing regular shapes from an image using contour analysis.

    Attributes
    ----------
    image : NumPy.Array
        Image containing shapes.
    image_path : str
        Path to the image file.

    Methods
    -------
    __init__(image_path: str) -> None:
        Constructor for the RegularShapeDetector class.
    find_shapes() -> dict:
        Detects regular shapes in the image using contour analysis and returns shape information.
    determine_shape(num_corners: int) -> str:
        Determines the shape name based on the number of corners.
    is_regular_shape(approx: np.ndarray) -> bool:
        Checks if the shape represented by the given corners is regular.
    circular_specification(approx: np.ndarray) -> tuple:
        Checks if the shape represented by the given corners is a circle or an oval.
    is_square(approx: np.ndarray) -> bool:
        Checks if the shape represented by the given corners is a square.
    save_results_as_df(path_to_save: str) -> None:
        Saves the detected shape data to a CSV file.
    display_image() -> None:
        Displays the original image using Matplotlib.
    """

    def __init__(self, image_path: str) -> None:
        """
        Constructor for the RegularShapeDetector class.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        None
        """
        self.image = cv2.imread(image_path)
        self.shape_info = self.find_shapes()


    def find_shapes(self):
        """
        Detects regular shapes in the image using contour analysis and returns shape information.

        Returns
        -------
        dict
            A dictionary containing information about the detected shapes.

        Notes
        -----
        The dictionary structure is as follows:
        {
            'Shape_i': {
                'Name': str,
                'Area': float,
                'Regular': bool
            }
        }

        """
        shape_info = {}
        edges = cv2.Canny(self.image, 50, 150)
        _, thresh = cv2.threshold(edges,
                                  230,
                                  255,
                                  cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            perimeter = cv2.arcLength(contour, True)
            # approx = coords for each corner
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # Colloct info abt shape:
            area = cv2.contourArea(contour)
            is_regular = self.is_regular_shape(approx)

            shape_name = self.determine_shape(len(approx)) # pass amnt of corners

            # Diffirentiate intersecting shapes:
            if shape_name == 'Rectangle' and self.is_square(approx):
                shape_name = 'Square'
                is_regular = True

            if shape_name == 'Circular':
                shape_name, is_regular =  self.circular_specification(approx)

            # Save and return info:
            shape_info[f"Shape_{str(i)}"] = {
                'Name': shape_name,
                'Area': area,
                'Regular': is_regular,
            }

        return shape_info

    def determine_shape(self, num_corners: int) -> str:
        """
        Determines the shape name based on the number of corners.

        Parameters
        ----------
        num_corners : int
            Number of corners in the shape.

        Returns
        -------
        str
            The name of the shape.
        """
        # Define shape names using a dictionary
        shape_names = {
            0: 'Circular',
            3: 'Triangle',
            4: 'Rectangle',
            5: 'Pentagon',
            6: 'Hexagon'
            # Add more shapes if needed
        }

        # Return the shape name if num corners is in the dictionary,
        # otherwise default to 'Circular'
        return shape_names.get(num_corners, 'Circular')

    def is_regular_shape(self, approx: np.ndarray) -> bool:
        """
        Checks if the shape represented by the given corners is regular.

        Parameters
        ----------
        approx : np.ndarray
            Array containing corner coordinates of the shape.

        Returns
        -------
        bool
            True if the shape is regular, False otherwise.
        """
        # Calculate the lengths of the sides
        side_lengths = []
        for i in range(len(approx)): # len of corners
            corner_1 = approx[i][0] # corner 1
            corner_2 = approx[(i + 1) % len(approx)][0] # next corner

            # To calc the distance between two corners based on coords,
            # we need to use the Euclidean Distance Formula: 
            # It is builtin method in numpy!! :D
            length = np.linalg.norm(corner_1 - corner_2)
            side_lengths.append(int(length))

        # Check if all sides are approximately equal
        tolerance = 5  # Pixel tolerance
        first_legnth = side_lengths[0]
        for length in side_lengths[1:]:
            if abs(length - first_legnth) > tolerance: # Compare to first legnth
                return False
        return True

    def circular_specification(self, approx: np.ndarray) -> tuple:
        """
        Checks if the shape represented by the given corners is a circle or an oval.

        Parameters
        ----------
        approx : np.ndarray
            Array containing corner coordinates of the shape.

        Returns
        -------
        tuple
            A tuple containing the shape name and regularity status.
        """
        x, y, w, h = cv2.boundingRect(approx)
        
        # Calculate aspect ratio n its threshold
        aspect_ratio = float(w) / h
        aspect_ratio_threshold = 0.2
        
        # If aspect ratio is close to 1, classify as circle, otherwise classify as oval
        # if circle it also defaults as regular if oval not
        if abs(aspect_ratio - 1) < aspect_ratio_threshold:
            return 'Circle', True
        else:
            return 'Oval', False

    def is_square(self, approx: np.ndarray) -> bool:
        """
        Checks if the shape represented by the given corners is a square.

        Parameters
        ----------
        approx : np.ndarray
            Array containing corner coordinates of the shape.

        Returns
        -------
        bool
            True if the shape is a square, False otherwise.
        """
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return True
        else:
            return False


    def save_results_as_df(self, path_to_save: str) -> None:
        """
        Saves the detected shape data to a CSV file.

        Parameters
        ----------
        path_to_save : str
            Path to save the CSV file.

        Returns
        -------
        None
        """
        df = pd.DataFrame.from_dict(self.shape_info, orient='index')
        df = df.sort_values(by='Area', ascending=False)
        df.to_csv(path_to_save, index=False)

    def display_image(self) -> None:
        """
        Displays the original image using Matplotlib.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(rgb_image, cmap=None)
        plt.axis('off')
        plt.show()

