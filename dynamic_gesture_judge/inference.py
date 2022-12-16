import os
import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data_generate import KeyPointsProcess
import mediapipe as mp
import numpy as np
from RNN_train import RNN

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def inference_point(img):
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        _image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(_image)
        hand_point = KeyPointsProcess(results, _image.shape)
        if hand_point is not None:
            return hand_point, results
        else:
            return None, None

def print_point(img, results):
    if result is not None:
        mp_drawing.draw_landmarks(
            img,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return img


def inference_gesture(point):
    seq_len = 10  # 数据在时间维度上的采样数量，例如一个句子有10个单词，其为10，一个视频数据有30帧，其为30
    num_hidden_layers = 2  # 隐藏层数
    hidden_size = 64  # 隐藏层单元数
    input_size = 42  # 单个数据在一个时刻的特征数
    num_classes = 3  # 分类个数
    label_ = {"right": 0, "small": 1, "up": 2}
    model = RNN(input_size, hidden_size, num_hidden_layers, len(label_))
    state_dict = torch.load("./model.pth")
    model.load_state_dict(state_dict['model'], strict=False)
    model = model.cuda()
    model.eval()
    point_tensor = torch.tensor([point])
    point_tensor = point_tensor.transpose(0, 1).cuda()
    ans = model(point_tensor)
    f = torch.nn.Softmax(dim=1)
    probability = f(ans)
    print(probability.tolist(), probability.argmax(dim=1))
    # max_class = probability.argmax(dim=0)
    # max_class = max_class.cpu()
    # if probability[max_class] >= 0.65:
    #     inf_res = label_[str(max_class.numpy().tolist())]
    # else:
    #     inf_res = "no gesture"
    # print(probability, inf_res)
    return ans

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    count = 0
    point_list = []
    while cap.isOpened():
        success, image = cap.read()
        # c_img = image.copy()
        img_point, result = inference_point(image)
        image = print_point(image, result)
        if img_point is not None:
            point_list.append(img_point.flatten())
        else:
            point_list.append(np.zeros(42, dtype=np.float32))
        count += 1
        # cv2.circle(image, (20, 20), 10, (0, 0, 255), -1)
        if count == 10:
            count = 0
            ans = inference_gesture(point_list)
            # print(ans)
            point_list.clear()
        cv2.imshow("camera", image)
        key = cv2.waitKey(50)
        if key == 27:
            break
    cv2.destroyAllWindows()
    cap.release()