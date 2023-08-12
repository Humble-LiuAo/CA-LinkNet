

# CA-LinkNet for Semantic segmentation of remote sensing images

该仓库包含CA-LinkNet 网络的 PyTorch 实现，用于多波段多类的遥感影像分割。

![image-20230812151504533](https://s2.loli.net/2023/08/12/5yoY4HsUK2EWtBL.png)

## 样本数据集

该网络可以在多波段遥感影像数据集上进行训练测试：

| 数据集 | 输入分辨率   |
| ------ | ------------ |
| train  | `25*256*256` |
| test   | `25*256*256` |
## Dependencies:

- Python 3.4 or greater
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)

## 文件/文件夹及其用途：

- main.py：主文件
- CA-LinkNet FCN Unet .py：网络架构实现
- data dataset dataPreprocess .py：数据预处理与增强
- model ：模型保存路径
- train.py：模型训练
- test.py ：计算测试误差
- evalutor.py：评估指标
- focalloss.py：焦点损失
- Repeated_predictionspy：忽略边缘预测大图
- plot.py：训练损失曲线

## 海雾识别结果

- 测试集不同算法的精度

| Models     | *MIoU*    | *POD*     | *FAR*     | *CSI*     | *HSS*     |
| ---------- | --------- | --------- | --------- | --------- | --------- |
| U-Net      | 0.862     | 0.868     | 0.106     | 0.788     | 0.848     |
| FCN        | 0.876     | 0.896     | 0.105     | 0.810     | 0.865     |
| LinkNet    | 0.883     | 0.871     | 0.068     | 0.819     | 0.873     |
| CA-LinkNet | **0.911** | **0.900** | **0.046** | **0.862** | **0.905** |

- 整景遥感影像分割结果

![image-20230812153200582](https://s2.loli.net/2023/08/12/nuzCryVkip8DZgt.png)
