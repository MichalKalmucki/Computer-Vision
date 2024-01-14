import cv2
from utils.object_detector import ObjectDetector

detector = ObjectDetector()

image = cv2.imread("resources/img1.jpg")
corners = detector.detect_corners(image)
print(corners)