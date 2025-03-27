from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QLabel,QVBoxLayout,QDialog
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QLabel,QVBoxLayout,QDialog
from Ui_beiyesi import Ui_MainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

class ImageDialog(QDialog):
    def __init__(self, img_path, parent=None):
        super(ImageDialog, self).__init__(parent)
        self.setWindowTitle("Image Dialog")

        layout = QVBoxLayout()
        label = QLabel(self)
        pixmap = QPixmap(img_path)
        label.setPixmap(QPixmap(img_path))
        label.setScaledContents(True)

        layout.addWidget(label)
        self.setLayout(layout)

class beiyesi_Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(beiyesi_Window, self).__init__()
        self.setupUi(self)
        self.horizontalSlider.valueChanged.connect(self.update_lr)
        pixmap = QPixmap('./res.jpg')
        self.lbl_img.setPixmap(pixmap)
        self.lbl_img.setScaledContents(True)#设置图像自动缩放
        self.btn_exit.clicked.connect(self.close)
        self.btn_train.clicked.connect(self.train_naive_bayes)
    
    def update_lr(self, value):
        self.lbl_lr.setText(str(0.01*value))

    def train_naive_bayes(self):
        try:
            dataset_name = self.comboBox_2.currentText()
            if dataset_name == 'Iris':
                data = load_iris()
            elif dataset_name == 'Digits':
                data = load_digits()
            else:
                raise Exception('Dataset not found')

            X = data.data
            y = data.target

            #划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            # 创建并训练朴素贝叶斯模型
            model = GaussianNB()
            model.fit(X_train, y_train)

            #预测
            y_pred = model.predict(X_test)

            #计算准确率
            accuracy = accuracy_score(y_test, y_pred)

            #显示结果
            result_text = f"Dataset：{dataset_name}\nAccuracy: {accuracy:.4f}\nPredictions: {y_pred}"
            self.textBrowser.setText(str(result_text))

            #结果可视化
            axs,fig = plt.subplots(1,2)
            fig[0].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis',marker = 'o')
            fig[0].set_title('Predicted')
            fig[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis',marker = 'o')
            fig[1].set_title('Actual')
            plt.show()
            plt.savefig('./result.jpg')
            self.lbl_img.setPixmap(QPixmap('./result.jpg'))

        except Exception as e:
            QMessageBox.warning(self, "错误", "{}".format(e))
        
    def on_lbl_img_clicked(self,event):
        if event.button() == Qt.LeftButton:
            dialog = ImageDialog('./result.jpg',self)
            dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = beiyesi_Window()
    win.show()
    sys.exit(app.exec_())