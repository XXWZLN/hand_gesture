# hand_gesture
## 简介
使用Google的mediapipe手部关键点检测模型，使用多层感知机训练手势分类，当前3分类准确率97%。
没有设置no gesture的分类。
## 使用流程
1. 数据集创建：用jsonDatasetGenerate.py生成关键点json文件，用csv_file_generate.py生成数据目录，在数据集根目录创建文件label.csv输入分类标签
2. 修改模型：修改softmax层输出节点数 == 分类数
3. 训练