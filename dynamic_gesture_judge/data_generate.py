import os
import csv
import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def KeyPointsProcess(results, img_shape):
    """
    手部关键点数据处理
    :param results: mediapipe给出的关键点检测结果
    :param img_shape: 原图形状
    :return: （21，2）的手部关键点坐标，以手部的宽高为分母，手部方框左上点为原点归一化
    """
    if results.multi_hand_landmarks:
        h, w, c = img_shape
        xmin, ymin = w, h
        xmax, ymax = 0, 0
        # numpy 数据类型要注意！！！
        hand_point = np.array([[int(res.x * w), int(res.y * h)] for res in results.multi_hand_landmarks[0].landmark],
                              dtype=np.float32)
        for id, res in enumerate(results.multi_hand_landmarks[0].landmark):
            if res.x < xmin:
                xmin = res.x
            if res.y < ymin:
                ymin = res.y
            if res.x > xmax:
                xmax = res.x
            if res.y > ymax:
                ymax = res.y
            hand_left_up = (int(xmin * w) - 5, int(ymin * h) - 5)
            hand_right_down = (int(xmax * w) + 5, int(ymax * h) + 5)
            hand_w = hand_right_down[0] - hand_left_up[0]
            hand_h = hand_right_down[1] - hand_left_up[1]
        for point in hand_point:
            point[0] = (point[0] - hand_left_up[0]) / hand_w
            point[1] = (point[1] - hand_left_up[1]) / hand_h
        return hand_point
    else:
        return None


def dataGenerate(pic):
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        _image = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        results = hands.process(_image)
        # print(results.multi_hand_landmarks[0])
        hand_point = KeyPointsProcess(results, _image.shape)
        if hand_point is not None:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].scatter(hand_point[:, 0], hand_point[:, 1])
            axes[0].xaxis.set_ticks_position('top')
            axes[0].invert_yaxis()
            # ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
            # ax.invert_yaxis()
            # plt.show()
            mp_drawing.draw_landmarks(
                pic,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            axes[1].imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
            axes[1].axis('off')
            plt.close()
            # fig.show()
            return hand_point, fig
        else:
            return None, None



if __name__ == "__main__":
    # 按键截取相机图片
    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     success, image = cap.read()
    #     cv2.imshow("camera", image)
    #     k = cv2.waitKey(1)
    #     if k == 27:
    #         pic = image
    #         break
    # cv2.destroyAllWindows()
    # cap.release()
    # figure = dataGenerate(pic)

    PATH_RAW = './dataset/raw_img/small/train'
    PATH_CSV = './dataset/keypoint/small/train'
    for signal_data_path in os.listdir(PATH_RAW):
        signal_data = PATH_RAW + "/" + signal_data_path
        with open(PATH_CSV + "/" + signal_data_path + '.csv', "w", newline='') as cf:
            writer = csv.writer(cf)
            for file in os.listdir(signal_data):
                pic = cv2.imread(signal_data + '/' + file)
                hand_point, figure = dataGenerate(pic)
                if figure is not None:
                    figure.savefig(signal_data + '/' + file.split('.')[0] + '_c.jpg')
                if hand_point is not None:
                    writer.writerow(hand_point.flatten())
                else:
                    writer.writerow(np.zeros(42, dtype=np.float32))