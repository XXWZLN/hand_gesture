from torch.utils.data import Dataset
import os
import json
import torch
import csv

# class LoadData(Dataset):
#     def __init__(self, path, train_or_test="train" ):
#         self.label_list = []
#         self.data_path = []
#         self.label_num = {}
#         with open(path + "/label.csv") as f_index:
#             lab_reader = csv.DictReader(f_index)
#             for dic_row in lab_reader:
#                 key, val = dic_row["label"], dic_row["idx"]
#                 self.label_num[key] = val
#
#         if train_or_test == "train":
#             with open(path + "/train/index.csv") as img_index_csv_f:
#                 img_index_csv = csv.DictReader(img_index_csv_f)
#                 for img_index_row in img_index_csv:
#                     img_label = img_index_row['label']
#                     self.label_list.append(self.label_num[img_label])
#
#
#
#         else:
#             with open(path + "/test/index.csv") as img_index_csv_f:
#                 img_index_csv = csv.DictReader(img_index_csv_f)
#                 for img_index_row in img_index_csv:
#                     img_label = img_index_row['label']
#                     self.data_path.append(img_index_row['data_path'])
#                     self.label_list.append(self.label_num[img_label])
#         # self.file_list = torch.tensor([])
#         # self.label_list = torch.tensor([])
#
#         # for labels in os.listdir(path):
#         #     if labels in label_.keys():
#         #         # label_idx = label_[labels]
#         #         # label_this = torch.zeros(1, label_.__len__())
#         #         # label_this[0][label_idx] = 1
#         #         if "train" == train_or_test:
#         #             for item_dir in os.listdir(path + "/" + labels + "/train"):
#         #                 self.label_list = torch.cat((self.label_list, torch.tensor([label_[labels]])), 0)
#         #
#         #                 with open(path + "/" + labels + "/train" + "/" + item_dir) as f:
#         #                     hand_dict_ = json.load(f)
#         #                 pl = []
#         #                 for point in hand_dict_.values():
#         #                     pl.append(point['x'])
#         #                     pl.append(point['y'])
#         #                 self.file_list = torch.cat((self.file_list, torch.tensor([pl])), 0)
#         #         else:
#         #             for item_dir in os.listdir(path + "/" + labels + "/test"):
#         #                 #self.label_list = torch.cat((self.label_list, label_this),0)
#         #                 self.label_list = torch.cat((self.label_list, torch.tensor([label_[labels]])), 0)
#         #                 with open(path + "/" + labels + "/test" + "/" + item_dir) as f:
#         #                     hand_dict_ = json.load(f)
#         #                 pl = []
#         #                 for point in hand_dict_.values():
#         #                     pl.append(point['x'])
#         #                     pl.append(point['y'])
#         #                 self.file_list = torch.cat((self.file_list, torch.tensor([pl])), 0)
#
#
#     def __len__(self):
#         return len(self.label_list)
#
#     def __getitem__(self, index):
#         img_label = self.label_list[index]
#         img_path = self.data_path[index]
#         with open(img_path) as f:
#             hand_dict_ = json.load(f)
#         if self.label_num[hand_dict_["label"]] != img_label:
#             print("buduiqi")
#         pl = []
#         point42 = hand_dict_["dataset"]
#         for n in range(21):
#             pl.append(point42[str(n)]["x"])
#         for n in range(21):
#             pl.append(point42[str(n)]["y"])
#         pt = torch.tensor(pl)
#         return pt, int(img_label)





# gititem解决之前
class LoadData(Dataset):
    def __init__(self, path, train_or_test="train" ):
        self.label_list = []
        self.data_path = []
        self.label_num = {}
        with open(path + "/label.csv") as f_index:
            lab_reader = csv.DictReader(f_index)
            for dic_row in lab_reader:
                key, val = dic_row["label"], dic_row["idx"]
                self.label_num[key] = val

        if train_or_test == "train":
            with open(path + "/train/index.csv") as img_index_csv_f:
                img_index_csv = csv.DictReader(img_index_csv_f)
                for img_index_row in img_index_csv:
                    img_label = img_index_row['label']
                    self.data_path.append(img_index_row['data_path'])
                    self.label_list.append(self.label_num[img_label])
        else:
            with open(path + "/test/index.csv") as img_index_csv_f:
                img_index_csv = csv.DictReader(img_index_csv_f)
                for img_index_row in img_index_csv:
                    img_label = img_index_row['label']
                    self.data_path.append(img_index_row['data_path'])
                    self.label_list.append(self.label_num[img_label])
        # self.file_list = torch.tensor([])
        # self.label_list = torch.tensor([])

        # for labels in os.listdir(path):
        #     if labels in label_.keys():
        #         # label_idx = label_[labels]
        #         # label_this = torch.zeros(1, label_.__len__())
        #         # label_this[0][label_idx] = 1
        #         if "train" == train_or_test:
        #             for item_dir in os.listdir(path + "/" + labels + "/train"):
        #                 self.label_list = torch.cat((self.label_list, torch.tensor([label_[labels]])), 0)
        #
        #                 with open(path + "/" + labels + "/train" + "/" + item_dir) as f:
        #                     hand_dict_ = json.load(f)
        #                 pl = []
        #                 for point in hand_dict_.values():
        #                     pl.append(point['x'])
        #                     pl.append(point['y'])
        #                 self.file_list = torch.cat((self.file_list, torch.tensor([pl])), 0)
        #         else:
        #             for item_dir in os.listdir(path + "/" + labels + "/test"):
        #                 #self.label_list = torch.cat((self.label_list, label_this),0)
        #                 self.label_list = torch.cat((self.label_list, torch.tensor([label_[labels]])), 0)
        #                 with open(path + "/" + labels + "/test" + "/" + item_dir) as f:
        #                     hand_dict_ = json.load(f)
        #                 pl = []
        #                 for point in hand_dict_.values():
        #                     pl.append(point['x'])
        #                     pl.append(point['y'])
        #                 self.file_list = torch.cat((self.file_list, torch.tensor([pl])), 0)


    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img_label = self.label_list[index]
        img_path = self.data_path[index]
        with open(img_path) as f:
            hand_dict_ = json.load(f)
        if self.label_num[hand_dict_["label"]] != img_label:
            print("buduiqi")
        pl = []
        point42 = hand_dict_["dataset"]
        for n in range(21):
            pl.append(point42[str(n)]["x"])
        for n in range(21):
            pl.append(point42[str(n)]["y"])
        pt = torch.tensor(pl)
        return pt, int(img_label)

