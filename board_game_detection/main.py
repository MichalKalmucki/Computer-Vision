import cv2
import numpy as np
from tqdm import tqdm
from utils.object_detector import ObjectDetector
from utils.template_matcher import TemplateMatcher

# function that receives a sequence of numbers and returns a list of the highest number in each sequence of consecutive numbers
def denoise_detections(sequence):
    highest_numbers = []
    current_number = None
    consecutive_count = 0
    highest_number = None

    for number in sequence:
        if number == current_number:
            consecutive_count += 1
        else:
            consecutive_count = 1
            current_number = number

        if consecutive_count >= 2:
            if highest_number is None or current_number > highest_number:
                highest_number = current_number
        highest_numbers.append(highest_number)

    return highest_numbers


def main():
    detector = ObjectDetector()
    matcher = TemplateMatcher()

    capitol= cv2.imread('templates/capitol_template.png')
    capitol_opp = cv2.imread('templates/capitol_opp_template.png')
    unit = cv2.imread('templates/unit_template.png')
    unit_opp = cv2.imread('templates/unit_opp_template.png')
    support = cv2.imread('templates/support_template.png')
    support_opp = cv2.imread('templates/support_opp_template.png')
    deck = cv2.imread('templates/deck_template.png')
    deck_opp = cv2.imread('templates/deck_opp_template.png')
    deck_shadow = cv2.imread('templates/deck_shadow_template.png')
    deck_opp_shadow = cv2.imread('templates/deck_opp_shadow_template.png')
    barrel = cv2.imread('templates/barrel_template.png')
    barrel_opp = cv2.imread('templates/barrel_opp_template.png')

    templates = [capitol, capitol_opp, 
                 unit, unit_opp, 
                 support, support_opp, 
                 deck, deck_opp, 
                 deck_shadow, deck_opp_shadow, 
                 barrel, barrel_opp]

    classes = ['Capitol', 'Opponent Capitol', 
               'Unit', 'Opponent Unit', 
               'Support', 'Opponent Support', 
               'Deck', 'Opponent Deck', 
               'Deck', 'Opponent Deck',  # Second time since we needed seperate templates for the deck with shadow
               'Barrel', 'Opponent Barrel']
    
    types = ['easy', 'medium', 'hard', 'very_hard']

    #for each processed frame there is a lot of class detections to classify events, more info on that in readme
    for type in types:
        for i in range(1, 4):
            video_path = f'resources/{type}{i}.mp4'
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("Error: Couldn't open video.")
                exit()

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_video_path = f'outputs/{type}{i}.avi'
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            frame_number = 0
            process_every_n_frames = 20
            display_counter = 999
            card_played = None
            deck_detected = [False, False]
            resources = [0, 0]
            cards = {
                'Unit': (0, []),
                'Support': (0, []),
                'Opponent Unit': (0, []),
                'Opponent Support': (0, [])
            }

            for frame_number in tqdm(range(frame_count), desc=f"Processing {type}{i}.mp4"):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_number % process_every_n_frames == 0:
                    detected_objects, detected_cords = detector.detect_objects(frame)

                    determined_classes = matcher.match_templates(templates, detected_objects)
                    deck_detected_this_frame = [False, False]
                    resources = [0, 0]
                    cards_this_frame = {
                        'Unit': 0,
                        'Support': 0,
                        'Opponent Unit': 0,
                        'Opponent Support': 0
                    }

                    for obj_class in determined_classes:
                        if (np.argmax(obj_class) == 8 or np.argmax(obj_class) == 6) and obj_class[np.argmax(obj_class)] > 0.25:
                            deck_detected[0] = True
                            deck_detected_this_frame[0] = True
                            continue

                        if (np.argmax(obj_class) == 9 or np.argmax(obj_class) == 7) and obj_class[np.argmax(obj_class)] > 0.25:
                            deck_detected[1] = True
                            deck_detected_this_frame[1] = True
                            continue

                        if np.argmax(obj_class) == 10 and obj_class[np.argmax(obj_class)] > 0.25:
                            resources[0] += 1
                            continue
                        
                        if np.argmax(obj_class) == 11 and obj_class[np.argmax(obj_class)] > 0.25:
                            resources[1] += 1
                            continue

                        if np.argmax(obj_class) == 2 and obj_class[np.argmax(obj_class)] > 0.25:
                            cards_this_frame['Unit'] += 1
                        
                        if np.argmax(obj_class) == 25 and obj_class[np.argmax(obj_class)] > 0.25:
                            cards_this_frame['Opponent Unit'] += 1
                        
                        if np.argmax(obj_class) == 4 and obj_class[np.argmax(obj_class)] > 0.25:
                            cards_this_frame['Support'] += 1
                        
                        if np.argmax(obj_class) == 5 and obj_class[np.argmax(obj_class)] > 0.25:
                            cards_this_frame['Opponent Support'] += 1

                    for card_type, card_num in cards_this_frame.items():
                        cards[card_type][1].append(card_num)
                    
                    if not deck_detected_this_frame[0]:
                        deck_detected[0] = False
                    if not deck_detected_this_frame[1]:
                        deck_detected[1] = False

                for card_type, (card_num, sequence) in cards.items():
                    if denoise_detections(sequence)[-1] and card_num < denoise_detections(sequence)[-1]:
                        if card_num:
                            display_counter = 0
                            card_played = card_type
                            cv2.putText(frame, f'{card_played} Was Played', (600, 800), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 105, 180), 2)

                        cards[card_type] = (denoise_detections(sequence)[-1], sequence)
                        
                    if display_counter < 4 * process_every_n_frames:
                        display_counter += 1
                        cv2.putText(frame, f'{card_played} Was Played', (600, 800), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 105, 180), 2)
                        break

                cv2.putText(frame, f"Player 1 Resources: {resources[0]}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Player 2 Resources: {resources[1]}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if not deck_detected[0]:
                    cv2.putText(frame, "Player 1 Draws Card!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not deck_detected[1]:
                    cv2.putText(frame, "Player 2 Draws Card!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                frame = matcher.draw_detections(frame, detected_cords, determined_classes, classes)

                out.write(frame)

                frame_number += 1


            cap.release()
            out.release()

            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()