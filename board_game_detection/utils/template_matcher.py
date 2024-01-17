import cv2
import numpy as np


#class for matching templates to classes and drawing detections on image
class TemplateMatcher:
    #method that receives a list of templates and a list of detected objects and returns for each object vector of similarities (using opencv's match templates) to each template
    def match_templates(self, templates, detected_objects):
        determined_classes = [[] for _ in range(len(detected_objects))]

        for j, template in enumerate(templates):
            for i, obj in enumerate(detected_objects):
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                template_resized = cv2.resize(template_gray, (obj.shape[:2][::-1]))
                obj_gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(obj_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                determined_classes[i].append(max_val)
                if j >= len(templates) -2:
                    if obj.shape[0] + obj.shape[1] > 120 or obj.shape[0] + obj.shape[1] < 80:
                        determined_classes[i][j] = 0
                if j < len(templates) - 2:
                    if obj.shape[0] + obj.shape[1] < 200:
                        determined_classes[i][j] = 0

        return determined_classes

    # method that receives detected objects and their classes and draws a rectangle around each object with its class
    def draw_detections(self, image, detected_cords, determined_classes, classes):
        original_image = image.copy()

        for i, obj in enumerate(detected_cords):
            x, y, width, height = obj
            class_num = np.argmax(determined_classes[i])
            obj_class = classes[np.argmax(determined_classes[i])]

            colors = [
                (0, 255, 0), (255, 0, 0), (0, 0, 255),
                (255, 255, 0), (0, 255, 255), (255, 0, 255), 
                (128, 128, 0), (0, 128, 128), (128, 0, 128), 
                (255, 128, 128,), (128, 255, 128), (120, 120, 120)
            ]
            color = colors[class_num]
            label = obj_class
            if determined_classes[i][class_num] > 0.25:
                if obj_class != 'Barrel' and obj_class != 'Opponent Barrel':
                    cv2.rectangle(original_image, (x, y), (x + width, y + height), color, 2)
                    cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return original_image
