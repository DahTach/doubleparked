from ultralytics import YOLO
import argparse
import imagesize
import cv2
import numpy as np
import os
import sys
from cars import Cars

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="model path")
parser.add_argument("-i", "--images", type=str, help="images folder path")

args = parser.parse_args()

if not args.model:
    print("No model specified, remember to specify a model with -m or --model")
    exit(0)

if not args.images:
    print("No images specified, remember to specify a directory with -i or --images")
    exit(0)

# Load a pretrained YOLOv8n model
model = YOLO(args.model)


def walk(root, extensions=(".jpg", ".jpeg", ".png")):
    images_paths = []

    for root, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                image_path = os.path.join(root, filename)
                images_paths.append(image_path)

    return images_paths


def infer(model, image, imgsz):
    try:
        return model.predict(
            image, device="mps", save=False, show=False, imgsz=imgsz, conf=0.5
        )
    except Exception as e:
        raise e


def exit(code):
    print("Exiting the program...")
    try:
        sys.exit(code)
    except SystemExit:
        os._exit(code)
    except Exception as e:
        print(f"Could not exit because of {e}\n Force quitting ...")
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)


def main():
    for image in walk(args.images):
        imgsz = imagesize.get(image)
        img = cv2.imread(image)
        predictions = infer(model, img, imgsz)

        cars = Cars(predictions)

        cars.double_parked()

        cars.draw_cars(img)

        try:
            cv2.imshow("image", img)
        except KeyboardInterrupt:
            exit(0)

        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord("q") or key == 3:
            exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(130)
