#Kmeans
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QLabel,QVBoxLayout,QDialog
from Ui_Kmeans import Ui_MainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
from sklearn.datasets import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

class Kmeans_Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Kmeans_Window, self).__init__()
        self.setupUi(self)
        self.horizontalSlider.valueChanged.connect(self.update_lr)
        pixmap = QPixmap('./res.jpg')
        self.lbl_img.setPixmap(pixmap)
        self.lbl_img.setScaledContents(True)#设置图像自动缩放
        self.btn_exit.clicked.connect(self.close)
        self.btn_train.clicked.connect(self.train_Kmeans)
    
    def update_lr(self, value):
        self.lbl_lr.setText(str(0.01*value))

    def train_Kmeans(self):
        try:
            dataset_name = self.comboBox_2.currentText()
            if dataset_name == 'Iris':
                data = load_iris()
            elif dataset_name == 'Digits':
                data = load_digits()
            else:
                raise Exception('Dataset not found')
            
            X= data.data
            y = data.target

            #标准化数据
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            #创建KMeans模型
            Kmeans = KMeans(n_clusters=3, random_state=42)
            Kmeans.fit(X_scaled)

            #获取聚类结果
            labels = Kmeans.labels_
            centroids = Kmeans.cluster_centers_

            #评估
            silhouette = silhouette_score(X_scaled, labels)
            ari = adjusted_rand_score(y, labels)

            #显示结果
            result_sil = f"Dataset：{dataset_name}\nsilhouette_score：{silhouette:.4f}\n"
            result_ari = f"Dataset：{dataset_name}\nadjusted_rand_score：{ari:.4f}\n"
            self.textBrowser.setText(str(result_sil) + str(result_ari))

            #可视化结果
            plt.figure(figsize=(10,6))
            plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='viridis')
            plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=200, linewidths=3, color='red')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('K-means Clustering Results')
            plt.show()

            plt.tight_layout()
            plt.savefig('./Kmeans_res.jpg')
            self.lbl_img.setPixmap(QPixmap('./Kmeans_res.jpg'))
        except Exception as e:
            QMessageBox.warning(self, 'Warning', str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Kmeans_Window()
    window.show()
    sys.exit(app.exec_())
