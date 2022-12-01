import cv2
import torch
from train import ModuleMy
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def result_config(_results, img_shape):
    lmJson = {}
    h, w, c = img_shape
    xmin, ymin = w, h
    xmax, ymax = 0, 0
    myhand = _results.multi_hand_landmarks[0]
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
    hand_w = hand_right_down[0] - hand_left_up[0]
    hand_h = hand_right_down[1] - hand_left_up[1]

    for coordinate in lmJson.values():
        coordinate["x"] = (coordinate["x"] - hand_left_up[0]) / hand_w
        coordinate["y"] = (coordinate["y"] - hand_left_up[1]) / hand_h

    return lmJson


model = ModuleMy()
state_dict = torch.load("./model.pth")
model.load_state_dict(state_dict['model'], strict=False)
model = model.cuda()
model.eval()

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        _image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(_image)
        h, w, c = image.shape

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        if results.multi_hand_landmarks:
            hand_points = result_config(results, (h, w, c))
            pt = []
            for point_index in range(21):
                pt.append(hand_points[str(point_index)]["x"])
            for point_index in range(21):
                pt.append(hand_points[str(point_index)]["y"])
            input = torch.tensor(pt)
            input = input.cuda()
            output = model(input)
            f = torch.nn.Softmax(dim=0)
            probability = f(output)
            max_class = probability.argmax(dim=0)
            max_class = max_class.cpu()
            label = {"0": "like", "1": "ok", "2": "love"}
            if probability[max_class] >= 0.65:
                inf_res = label[str(max_class.numpy().tolist())]
            else:
                inf_res = "no gesture"
            print(probability, inf_res)



        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
