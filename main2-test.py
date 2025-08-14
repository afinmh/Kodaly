from copyreg import pickle
import mediapipe as mp
import cv2

import pickle

import time
import copy
import itertools

def load_svm_model():
    with open(r"model/Right_HandSM5.model", "rb") as file:
        model = pickle.load(file)
    return model

# Load class names
def load_class_names():
    f = open(r"gestures.names", 'r')

    class_names = f.read().split('\n')
    f.close()
    print(class_names)
    return class_names


def run_real_time_demo(cap, class_names, count_fps=False):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    model = load_svm_model()

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        estimate_pose = True
        previous_frame_time = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Camera Error")
            
            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Flip on horizontal (Mirroring Camera)
            image = cv2.flip(image, 1)
            
            # Detections Mediapipe
            results = hands.process(image)
            
            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            classNada = ""
            
            # Rendering results
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    mp_drawing.draw_landmarks(image,
                                              hand_landmarks,
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(250, 44, 250), 
                                                                     thickness=2, 
                                                                     circle_radius=2),
                                              )

                    hand_type = handedness.classification[0].label
                    print(f"Tangan: {hand_type}")

                    # Landmark calculation
                    landmark_list = calc_landmark_list(image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)

                    # NADA CLASSIFICATION USING SVM MODEL
                    prediction = model.predict([pre_processed_landmark_list])

                    classNada = class_names[prediction[0]]
                    print(f"Predicted Gesture Nada: {classNada}")

            
            # FPS counter
            if count_fps:
                current_frame_time = time.time()
                fps = 1/(current_frame_time - previous_frame_time)
                previous_frame_time = current_frame_time
                fps = int(fps)
                fps = str(fps)
                cv2.putText(image, "FPS:" + fps, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (128, 128, 128), 1, cv2.LINE_AA)

            if estimate_pose:
                cv2.putText(image, "Nada: " + classNada, (5,55), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (5,0,255), 2, cv2.LINE_AA)

            cv2.imshow("Hand Tracking And Gesture Recognition", image)

            key = cv2.waitKey(1)
            if key == 32: # SPACE
                estimate_pose = not estimate_pose
            if key == 27: # ESCAPE
                break

    cap.release()
    cv2.destroyAllWindows()


# Menentukan koordinat pixel landmarks pada gambar
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


if __name__ == "__main__":
    try:
        cap = cv2.VideoCapture(0)
        run_real_time_demo(cap, load_class_names(), count_fps=True)
    except Exception as e:
        print("Terjadi error:", e)
