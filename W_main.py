from PyQt5.QtWidgets import QApplication, QMainWindow
from Ui_main import Ui_MainWindow
from W_lr import LRWindow
from W_beiyesi import beiyesi_Window
from W_LinearRegression import LinearRegression_Window
from W_Kmeans import Kmeans_Window
from W_PCA import PCA_Window
from W_cal import cal_Window
from W_KNN_Classify import KNN_Classify_Window
from W_KNN_Regression import KNN_Regression_Window
from W_DBSCAN import DBSCAN_Window
from W_DecisionTree_Classify import DecisionTree_Classify_Window
from W_DecisionTree_Regression import DecisionTree_Regression_Window
from W_SVD import SVD_Window
from W_Gradient_Descent import Gradient_Descent_Window
import sys


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.action.triggered.connect(self.show_lr)
        self.action_2.triggered.connect(self.show_beiyesi)
        self.action_5.triggered.connect(self.show_LinearRegression)
        self.actionKmeans.triggered.connect(self.show_Kmeans)
        self.action_6.triggered.connect(self.show_PCA)
        self.action_7.triggered.connect(self.cal)
        self.action_KNN_C.triggered.connect(self.show_KNN_Classify)
        self.action_KNN_R.triggered.connect(self.show_KNN_Regression)
        self.action_DBSCAN.triggered.connect(self.show_DBSCAN)
        self.action_DecisionTree_Classify.triggered.connect(self.show_DecisionTree_Classify)
        self.action_DecisionTree_Regression.triggered.connect(self.show_DecisionTree_Regression)
        self.action_3.triggered.connect(self.show_Gradient_Descent)
        self.action_SVD.triggered.connect(self.show_SVD)

     #逻辑回归   
    def show_lr(self):
        self.win = LRWindow()
        self.win.show()

    #朴素贝叶斯
    def show_beiyesi(self):
        self.win = beiyesi_Window()
        self.win.show()

    #线性回归
    def show_LinearRegression(self):
        self.win = LinearRegression_Window()
        self.win.show()

    #Kmeans聚类   
    def show_Kmeans(self):
        self.win = Kmeans_Window()
        self.win.show()

    #主成分分析
    def show_PCA(self):
        self.win = PCA_Window()
        self.win.show()

    #KNN分类
    def show_KNN_Classify(self):
        self.win = KNN_Classify_Window()
        self.win.show()

    #KNN回归
    def show_KNN_Regression(self):
        self.win = KNN_Regression_Window()
        self.win.show()

    #DecisionTree分类
    def show_DecisionTree_Classify(self):
        self.win = DecisionTree_Classify_Window()
        self.win.show()

    #DecisionTree回归
    def show_DecisionTree_Regression(self):
        self.win = DecisionTree_Regression_Window()
        self.win.show()

    #DBSCAN聚类
    def show_DBSCAN(self):
        self.win = DBSCAN_Window()
        self.win.show()

    #SVD降维
    def show_SVD(self):
        self.win = SVD_Window()
        self.win.show()

    #梯度下降
    def show_Gradient_Descent(self):
        self.win = Gradient_Descent_Window()
        self.win.show()

    #简易计算器
    def cal(self):
        self.win = cal_Window()
        self.win.show()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.setStyleSheet("#MainWindow{border-image:url(bg.jpg)}")
    win.show()
    sys.exit(app.exec_())