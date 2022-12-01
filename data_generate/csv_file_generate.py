import csv
import os

def csv_file_generate(path):
    with open(path + "/index.csv", "w", newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(["data_path", "label"])
        for gesture in os.listdir(path):
            if os.path.splitext(gesture)[1] != ".csv":
                for img_path in os.listdir(path + "/" + gesture):
                    if os.path.splitext(img_path)[1] == ".json":
                        writer.writerow([path + "/" + gesture + "/" + img_path, gesture])

if __name__ == "__main__":
    train_path = "C:/Users/antarctic polar bear/Desktop/two_hand/train"
    test_path = "C:/Users/antarctic polar bear/Desktop/two_hand/test"

    csv_file_generate(train_path)
    csv_file_generate(test_path)