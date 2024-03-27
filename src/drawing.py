import cv2
import numpy as np


class Draw:
    def __init__(self) -> None:
        pass

    def line(self, img, line, thickness=2):
        red = 200
        green = 0
        blue = 0
        cv2.line(img, line[0], line[1], color=(blue, green, red), thickness=thickness)

    def rectangle(
        self,
        img,
        start_point,
        end_point,
        text=None,
        text_color=(255, 255, 255),
        color=(0, 0, 0),
        transparent=False,
    ):
        cv2.rectangle(img, start_point, end_point, color=color, thickness=2)
        if text is not None:
            cv2.putText(
                img,
                text,
                (int(start_point[0]), int(start_point[1]) - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=text_color,
                thickness=2,
            )
        if transparent:
            self._transparent(img, start_point, end_point, color=(0, 0, 255))

    def _transparent(
        self,
        img,
        start_point,
        end_point,
        color=(0, 0, 255),
        shape="rectangle",
        radius=0,
    ) -> None:
        # Initialize blank mask image of same dimensions for drawing the shapes
        shapes = np.zeros_like(img, np.uint8)

        # Draw shapes
        if shape == "rectangle":
            cv2.rectangle(shapes, start_point, end_point, color, cv2.FILLED)
        if shape == "circle":
            cv2.circle(shapes, start_point, radius, color, cv2.FILLED)

        # Generate output by blending image with shapes image
        alpha = 0.6
        mask = shapes.astype(bool)
        img[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]
