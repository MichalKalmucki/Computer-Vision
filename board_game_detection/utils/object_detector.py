import cv2
import numpy as np

class ObjectDetector:
    def perspective_transform(self, image, corners):
        ordered_corners = corners
        bottom_l, bottom_r, top_r, top_l = ordered_corners

        width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
        width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
        width = max(int(width_A), int(width_B))

        height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
        height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
        height = max(int(height_A), int(height_B))

        dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                        [0, height - 1]], dtype = "float32")

        ordered_corners = np.array(ordered_corners, dtype="float32")

        matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

        return cv2.warpPerspective(image, matrix, (width, height))

    def find_extreme_points(self, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        bottom_left = tuple(box[0])
        top_left = tuple(box[3])
        bottom_right = tuple(box[1])
        top_right = tuple(box[2])

        return bottom_left, bottom_right, top_right, top_left

    def detect_corners(self, image):
        image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel_size = (5, 5)

        blurred_image = cv2.blur(gray, kernel_size)

        sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = np.uint8(magnitude)

        _, thresh = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)
        corners = self.find_extreme_points(largest_contour)

        return np.array(corners)
    
    def detect_objects(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)

        sobel_combined = np.uint8(sobel_combined)

        _, binary_image = cv2.threshold(sobel_combined, np.mean(sobel_combined) * 2, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 4_000]

        detected_objects = []
        detected_cords = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            detected_cords.append((x, y, w, h))
            cropped_image = image[y:y + h, x:x + w]
            detected_objects.append(cropped_image)

        return detected_objects, detected_cords