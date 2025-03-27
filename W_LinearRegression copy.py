from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QLabel,QVBoxLayout,QDialog
from Ui_LinearRegression import Ui_MainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
class LinearRegression_Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(LinearRegression_Window, self).__init__()
        self.setupUi(self)
        self.horizontalSlider.valueChanged.connect(self.update_lr)
        pixmap = QPixmap('./res.jpg')
        self.lbl_img.setPixmap(pixmap)
        self.lbl_img.setScaledContents(True)#设置图像自动缩放
        self.btn_exit.clicked.connect(self.close)
        self.btn_train.clicked.connect(self.train_LinearRegression)
    
    def update_lr(self, value):
        self.lbl_lr.setText(str(0.01*value))

    def train_LinearRegression(self):
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

            # 创建并训练逻辑回归模型
            model = LinearRegression()
            model.fit(X_train, y_train)

            #预测
            y_pred = model.predict(X_test)

            #计算均方误差
            mse = mean_squared_error(y_test,y_pred)

            #显示结果
            result_text = f"Dataset：{dataset_name}\nmse: {mse:.4f}\nPredictions: {y_pred}"
            self.textBrowser.setText(str(result_text))

            # 预测值散点图
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].scatter(range(len(y_test)), y_pred, color='blue', label='Predicted')
            axs[0, 0].scatter(range(len(y_test)), y_test, color='red', label='Actual')
            axs[0, 0].set_xlabel('Sample Index')
            axs[0, 0].set_ylabel('Value')
            axs[0, 0].set_title('Predicted vs Actual')
            axs[0, 0].legend()

            # 特征散点图
            axs[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o')
            axs[0, 1].set_title('Predicted')
            axs[0, 1].set_xlabel('Feature 1')
            axs[0, 1].set_ylabel('Feature 2')

            # 残差图
            residuals = y_test - y_pred
            axs[1, 0].scatter(range(len(residuals)), residuals, color='green')
            axs[1, 0].axhline(y=0, color='black', linestyle='--')
            axs[1, 0].set_xlabel('Sample Index')
            axs[1, 0].set_ylabel('Residuals')
            axs[1, 0].set_title('Residuals Plot')

            # 模型系数图
            coefficients = model.coef_
            feature_names = [f'Feature {i+1}' for i in range(len(coefficients))]
            axs[1, 1].bar(feature_names, coefficients, color='purple')
            axs[1, 1].set_xlabel('Features')
            axs[1, 1].set_ylabel('Coefficients')
            axs[1, 1].set_title('Model Coefficients')

            plt.show()
            plt.tight_layout()
            plt.savefig('./result.jpg')
            self.lbl_img.setPixmap(QPixmap('./result.jpg'))

        except Exception as e:
            QMessageBox.warning(self, "错误", "{}".format(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LinearRegression_Window()
    win.show()
    sys.exit(app.exec_())