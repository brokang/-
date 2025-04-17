import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap

# 加载预训练的 ResNet50 模型
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
# 修改模型结构
num_ftrs = model.fc.in_features
# 假设你有 6 个类别
model.fc = nn.Linear(num_ftrs, 6)

# 加载训练好的模型参数
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义类别标签映射表
class_labels = {
    0: 'cardboard',
    1: 'glass',
    2: 'mental',  # 这里可以根据实际情况修改
    3: 'paper',
    4: 'paper',
    5: 'trash'
}


def predict_image(image_path):
    try:
        # 打开图片
        image = Image.open(image_path)
        # 预处理图片
        image = transform(image).unsqueeze(0)

        # 使用模型进行预测
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)

        return predicted.item()
    except Exception as e:
        print(f"预测时出错: {e}")
        return None


class ImagePredictionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 创建选择图片按钮
        self.select_button = QPushButton('选择图片', self)
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        # 创建显示图片的标签
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # 创建显示预测结果的标签
        self.result_label = QLabel(self)
        layout.addWidget(self.result_label)

        self.setLayout(layout)
        self.setWindowTitle('图片分类预测')
        self.show()

    def select_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, '选择图片', '', '图像文件 (*.png *.jpg *.jpeg)')

        if file_path:
            # 显示图片
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(224, 224))

            # 进行预测
            result = predict_image(file_path)
            if result is not None:
                label = class_labels.get(result, '未知类别')
                self.result_label.setText(f"预测结果: {label}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ImagePredictionGUI()
    sys.exit(app.exec_())
