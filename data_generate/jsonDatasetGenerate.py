import os

import cv2
import mediapipe as mp
import numpy as np
import json

path = "F:/Light-HaGRID/md"
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands

def KeyPointsGenerate(_img):
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        image = cv2.flip(_img, 1)
        # Convert the BGR image to RGB before processing.
        _results = hands.process(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB))
        h, w, c = image.shape
    return _results, (h,w,c)

# for gesture in os.listdir(path):
#     for tov in os.listdir(path + "/" + gesture):
#         data_num = 0
#         for img in os.listdir(path + "/" + gesture + "/" + tov):
#             imgPath = path + "/" + gesture + "/" + tov + "/" + img
#             results, img_shape = KeyPointsGenerate(imgPath)
#             if results.multi_hand_landmarks:
#                 lmJson = {}
#                 h, w, c = img_shape
#                 xmin, ymin = w, h
#                 xmax, ymax = 0, 0
#                 myhand = results.multi_hand_landmarks[0]
#                 for id, lm in enumerate(myhand.landmark):
#                     if lm.x < xmin:
#                         xmin = lm.x
#                     if lm.y < ymin:
#                         ymin = lm.y
#                     if lm.x > xmax:
#                         xmax = lm.x
#                     if lm.y > ymax:
#                         ymax = lm.y
#                     cx, cy = int(lm.x * w), int(lm.y * h)
#                     lmJson[str(id)] = {"x": cx, "y": cy}
#                 hand_left_up = (int(xmin * w) - 5, int(ymin * h) - 5)
#                 hand_right_down = (int(xmax * w) + 5, int(ymax * h) + 5)
#                 hand_w = hand_left_up[0] + hand_right_down[0]
#                 hand_h = hand_left_up[1] + hand_right_down[1]
#
#                 for coordinate in lmJson.values():
#                     coordinate["x"] = (coordinate["x"] - hand_left_up[0]) / hand_w
#                     coordinate["y"] = (coordinate["y"] - hand_left_up[1]) / hand_h
#
#                 data_num += 1
#
#                 with open("./dataset/" + gesture + "/" + tov + "/" + str(data_num) + ".json" , "w") as f:
#                     json.dump(lmJson, f, indent=2)



def img_point_g(imgPath, label):
    # 单片演示
    vis = 0
    image = cv2.imread(imgPath)
    results, img_shape = KeyPointsGenerate(image)
    if results.multi_hand_landmarks:
        lmJson = {}
        h, w, c = img_shape
        xmin, ymin = w, h
        xmax, ymax = 0, 0
        myhand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(myhand.landmark):
            if lm.x < xmin:
                xmin = lm.x
            if lm.y < ymin:
                ymin = lm.y
            if lm.x > xmax:
                xmax = lm.x
            if lm.y > ymax:
                ymax = lm.y
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmJson[str(id)] = {"x": cx, "y": cy}
        hand_left_up = (int(xmin * w) - 5, int(ymin * h) - 5)
        hand_right_down = (int(xmax * w) + 5, int(ymax * h) + 5)
        hand_w =  hand_right_down[0] - hand_left_up[0]
        hand_h = hand_right_down[1] - hand_left_up[1]

        for coordinate in lmJson.values():
            coordinate["x"] = (coordinate["x"] - hand_left_up[0]) / hand_w
            coordinate["y"] = (coordinate["y"] - hand_left_up[1]) / hand_h

        data = {"label":label, "dataset":lmJson}

        # 在图像中显示点
        mp_drawing.draw_landmarks(
            image,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        img = np.zeros((100, 100), np.uint8)
        cv2.circle(image, hand_left_up, int(w / 50), (255, 0, 255), cv2.FILLED)
        for point in lmJson.values():
            cv2.circle(img, (int(point["x"] * hand_w), int(point["y"] * hand_h)), 3, (255, 0, 255), cv2.FILLED)
        if vis:
            cv2.namedWindow("hand")
            cv2.imshow("hand", img)
            cv2.imshow("h",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return (image, img, data)
    return None

def file_generate(raw_path, save_path):
    for gesture in os.listdir(raw_path):
        data_num = 0
        for img_path in os.listdir(raw_path + "/" + gesture):
            imgPath = raw_path + "/" + gesture + "/" + img_path
            re = img_point_g(imgPath, gesture)
            if re is None:
                continue
            data_num += 1
            raw_img, point_img, point = re
            filename = save_path + "/" + gesture + "/" + gesture + "_" + str(data_num)
            with open(filename + ".json", "w") as f:
                json.dump(point, f, indent=2)
            cv2.imwrite(filename + "_raw.jpg", raw_img)
            cv2.imwrite(filename + "_point.jpg", point_img)

if __name__ == '__main__':
    test_path = "C:/Users/antarctic polar bear/Desktop/two_hand_raw/test"
    train_path = "C:/Users/antarctic polar bear/Desktop/two_hand_raw/train"
    train_save_path = "C:/Users/antarctic polar bear/Desktop/two_hand/train"
    test_save_path = "C:/Users/antarctic polar bear/Desktop/two_hand/test"

    file_generate(test_path, test_save_path)
    file_generate(train_path, train_save_path)
