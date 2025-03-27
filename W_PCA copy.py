#Kmeans
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QLabel,QVBoxLayout,QDialog
from Ui_PCA import Ui_MainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

class PCA_Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(PCA_Window, self).__init__()
        self.setupUi(self)
        self.horizontalSlider.valueChanged.connect(self.update_lr)
        pixmap = QPixmap('./res.jpg')
        self.lbl_img.setPixmap(pixmap)
        self.lbl_img.setScaledContents(True)#设置图像自动缩放
        self.btn_exit.clicked.connect(self.close)
        self.btn_train.clicked.connect(self.pca_run)
    
    def update_lr(self, value):
        self.lbl_lr.setText(str(value))

    def pca_run(self):
        try:
            dataset_name = self.comboBox_2.currentText()
            if dataset_name == 'Iris':
                data = load_iris()
            elif dataset_name == 'Digits':
                data = load_digits()
            else:
                raise Exception('Dataset not found!')
            
            X = data.data
            y = data.target

            #数据标准化
            scaler = StandardScaler()
            X_scaler = scaler.fit_transform(X)

            #创建PCA模型，指定要留住的主成分数
            pca = PCA(n_components=self.horizontalSlider.value())

            #拟合PCA模型
            X_pca = pca.fit_transform(X_scaler)

            #输出解释的方差比率
            explained_variance = pca.explained_variance_ratio_

            #显示结果
            explained_variance_str = ''.join(map(str, explained_variance))
            res = f"Dataset: {dataset_name}\nexplained_variance_ratio_:{explained_variance_str}\n"
            self.textBrowser.setText(str(res))

            #可视化结果
            plt.figure(figsize=(8, 6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis',edgecolors='k',alpha=0.7)
            plt.xlabel('First principal component')
            plt.ylabel('Second principal component')
            plt.title('PCA of Iris dataset')
            plt.colorbar()
            plt.show()

            plt.tight_layout()
            plt.savefig('./PCA_res.jpg')
            self.lbl_img.setPixmap(QPixmap('./PCA_res.jpg'))
        except Exception as e:
            QMessageBox.warning(self, 'Error', str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PCA_Window()
    window.show()
    sys.exit(app.exec_())