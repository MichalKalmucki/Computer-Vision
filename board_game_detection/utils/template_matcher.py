import cv2
import numpy as np


class TemplateMatcher:
    def match_templates(self, templates, detected_objects):
        matches = []
        determined_classes = [[] for _ in range(len(detected_objects))]
        for template in templates:
            for i, obj in enumerate(detected_objects):
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                template_resized = cv2.resize(template_gray, (obj.shape[:2][::-1]))
                obj_gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(obj_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                determined_classes[i].append(max_val)
                matches.append(max_val)

        return determined_classes
    
    def draw_detections(self, image, detected_cords, determined_classes, classes):
        original_image = image.copy()
        for i, obj in enumerate(detected_cords):
            x, y, width, height = obj
            class_num = np.argmax(determined_classes[i])
            obj_class = classes[np.argmax(determined_classes[i])//2]

            colors = [
                (0, 255, 0), (255, 0, 0), (0, 0, 255), 
                (255, 255, 0), (0, 255, 255), (120, 120, 120),
                (255, 0, 255), (128, 128, 0), (0, 128, 128),
                (128, 0, 128)
            ]
            color = colors[class_num]
            # Add text label
            if determined_classes[i][class_num] > 0.2:
                label = obj_class
            else:
                label = 'Not classified'
                color = colors[-1]

            cv2.rectangle(original_image, (x, y), (x + width, y + height), color, 2)
            cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return original_image