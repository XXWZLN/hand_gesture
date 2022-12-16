# hand_gesture
## static_gesture_judge
使用Google的mediapipe手部关键点检测模型，使用多层感知机训练手势分类，当前3分类准确率97%。
没有设置no gesture的分类。

### 使用流程
1. 数据集创建：用jsonDatasetGenerate.py生成关键点json文件，用csv_file_generate.py生成数据目录，在数据集根目录创建文件label.csv输入分类标签
2. 修改模型：修改softmax层输出节点数 == 分类数
3. 训练

## dynamic_gesture_judge

使用Google的mediapipe手部关键点检测模型，使用两层LSTM训练动态手势分类，demo训练了缩小、右划、上划三个手势（类似华为隔空手势控制手机的那个样子），训练集准确率99%，测试集准确率93%。

此demo包含数据集制作脚本`video_collection.py`和`data_generate.py`。前者可以录制手势视频，并将视频长度切割为所需的长度，视频中的每一帧以jpg格式存储。后者调用mediapip手部关键点检测模型，将视频中每一帧的手部关键点保存在csv文件中，同时生成可视化的手部关键点图像，均保存于dataset。

但是最终调用训练的模型的预测效果不太好。因为没有设置没有检测到手势的分类。



