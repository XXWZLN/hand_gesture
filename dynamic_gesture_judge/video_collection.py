import cv2
import os

PATH = './dataset/raw_img/up/test'



def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        if os.listdir(path) is not None:
            for file in os.listdir(path):
                os.remove(path + '/' + file)



if __name__ == "__main__":
    # 删除文件夹文件
    # if os.listdir(PATH) is not None:
    #     for file in os.listdir(PATH):
    #         os.remove(PATH + '/' + file)
    cap = cv2.VideoCapture(0)
    count = 0
    data_num = 0
    img_list = []
    while cap.isOpened():
        success, image = cap.read()
        c_img = image.copy()
        if count != 0:
            if count == 11:
                count = 0
                while 1:
                    for img in img_list:
                        cv2.imshow("judge", img)
                        cv2.waitKey(50)
                    k = cv2.waitKey(0)
                    if k == ord('y'):
                        data_num += 1
                        makedir(PATH + '/' + str(data_num))
                        for idx, img in enumerate(img_list):
                            cv2.imwrite(PATH + '/' + str(data_num) + '/' + str(idx) + '.jpg', img)
                        img_list.clear()
                        cv2.destroyWindow("judge")
                        break
                    elif k == ord('r'):
                        continue
                    elif k == ord('n'):
                        img_list.clear()
                        cv2.destroyWindow("judge")
                        break
            else:
                img_list.append(c_img)
                count += 1
                cv2.circle(image, (20,20), 10, (0,0,255), -1)
        cv2.imshow("camera", image)
        key = cv2.waitKey(50)
        if key == ord(" "):
            count = 1
        if key == 27:
            pic = image
            break
    cv2.destroyAllWindows()
    cap.release()
