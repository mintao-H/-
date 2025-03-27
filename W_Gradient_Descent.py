from PyQt5.QtWidgets import QVBoxLayout,QApplication, QGraphicsPixmapItem,QMainWindow, QMessageBox,QMenu,QAction,QFileDialog,QDialog,QGraphicsView, QGraphicsScene
from Ui_Gradient_Descent import Ui_MainWindow
from PyQt5.QtGui import QPixmap,QPainter
from PyQt5.QtCore import Qt
import sys
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ui_add_page import Ui_Dialog
from functools import partial
from Ui_Image_Viewer import Ui_Image_show
import os
import random
import string
class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super(CustomGraphicsView, self).__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor

        self.scale(zoomFactor, zoomFactor)
class Gradient_Descent_Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Gradient_Descent_Window, self).__init__()
        self.setupUi(self)
        pixmap = QPixmap(None)
        self.result_path = ''
        self.dataset_name.setText(None)
        self.lbl_img.setPixmap(pixmap)
        self.lbl_img.setScaledContents(True)#设置图像自动缩放
        self.btn_train.clicked.connect(self.train_lr)
        # 连接 lbl_img 的点击事件
        self.lbl_img.mousePressEvent = self.show_image_popup
        self.gradient_descent_testsize.setValue(0.1)
        self.learning_rate.setValue(0.01)
        self.method.setCurrentText('批梯度下降')
        self.epochs.setValue(100)
        self.batch_size.setValue(32)
        self.btn_openFile.clicked.connect(self.open_file_dialog)
        self.btn_exit.clicked.connect(self.close)
        #初始化按钮
        self.selected_group = None #用于跟踪当前选中的组
        self.selected_button = None  #用于跟踪当前选中的按钮
        self.groups = {}
        self.buttons = []
        self.page_states = {} # 用于存储每个页面的状态
        self.group_states = {} # 用于存储每个组别下的页面按钮状态
        #分组按钮
        self.group_buttons = ['pushButton_A', 'pushButton_B', 'pushButton_C', 'pushButton_D',
                           'pushButton_E', 'pushButton_F', 'pushButton_G', 'pushButton_H']
        for button_name in self.group_buttons:
            button = getattr(self, button_name, None)
            if button:
                self.groups[button] = []
                self.group_states[button] = {'buttons':[],
                                             'textBrowser':'',
                                             'lbl_img':'',
                                             'dataset':'',
                                             'gradient_descent_testsize':0.1,
                                             'method':'批梯度下降',
                                             'learning_rate':0.01,
                                             'epochs':100,
                                             'batch_size':32,
                                             'image_path':'',
                                             'last_used_button': None} #初始化组别状态
                button.clicked.connect(partial(self.select_group, button))
                self.buttons.append(button)
        #页面按钮
        self.page_buttons = ['pushButton_9','pushButton_10','pushButton_11','pushButton_12',
                           'pushButton_13','pushButton_14','pushButton_15','pushButton_16']
        for button_name in self.page_buttons:
            button = getattr(self, button_name, None)
            if button:
                button.pressed.connect(self.on_button_pressed)
                button.released.connect(self.on_button_released)
                if button.text() != '[Blank Page]':
                    button.clicked.connect(partial(self.select_page,button))
                button.setContextMenuPolicy(Qt.CustomContextMenu)
                button.customContextMenuRequested.connect(self.show_context_menu)
                self.buttons.append(button)
                self.page_states[button] = {'buttons':[],
                                             'textBrowser':'',
                                             'lbl_img':'',
                                             'dataset':'',
                                             'gradient_descent_testsize':0.1,
                                             'method':'批梯度下降',
                                             'learning_rate':0.01,
                                             'epochs':100,
                                             'batch_size':32,
                                             'image_path':'',
                                            }

        #初始化每个组别的页面按钮状态
        for group_button in self.groups.keys():
            state = {'buttons':[], 
                     'textBrowser': self.textBrowser.toPlainText(),
                     'lbl_img':'',
                     'dataset':'',
                     'gradient_descent_testsize':0.1,
                     'method':'批梯度下降',
                     'learning_rate':0.01,
                     'epochs':100,
                     'batch_size':32,
                     'image_path': ''
                     #'last_used_button': self.selected_button #保存最后一次使用的按钮
            }
            if group_button != self.selected_button:
                state['last_used_button'] = self.selected_button
                for btn in self.buttons:
                    state['buttons'].append((btn,btn.text(),btn.isEnabled()))
                self.group_states[group_button] = state
        #自动选择第一个组别
        if self.groups:
            first_group = next(iter(self.groups))
            first_group.setStyleSheet("background-color: yellow;")
            self.select_group(first_group)
    def show_image_popup(self,event):
        dialog = QDialog()
        ui = Ui_Image_show()
        ui.setupUi(dialog)
        #创建一个CustomGraphicsView 和 QGraphicsScene 来显示图像
        scene = QGraphicsScene()
        view = CustomGraphicsView(scene)
        layout = QVBoxLayout(dialog)
        layout.addWidget(view)
        #获取当前显示的图像
        pixmap = self.lbl_img.pixmap()
        if not pixmap.isNull():
            item = QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        dialog.exec_()


    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_path,_ = QFileDialog.getOpenFileName(self, "选择数据集文件","","ALL FILES (*);;CSV FILES (*.csv);;Text Files (*.txt);;Excel Files (*.xlsx)",options=options)
        if file_path:
            self.dataset_name.setText(file_path.split('/')[-1])

    def batch_gradient_descent(self,X, y, learning_rate, epochs):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        
        for _ in range(epochs):
            predictions = X.dot(theta)
            errors = predictions - y
            gradient = (1/m) * X.T.dot(errors)
            theta -= learning_rate * gradient
            
            cost = (1/(2*m)) * np.sum(errors**2)
            cost_history.append(cost)
            
        return theta

    def stochastic_gradient_descent(self,X, y, learning_rate, epochs):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        for _ in range(epochs):
            for i in range(m):
                xi = X[i:i+1]
                yi = y[i:i+1]
                prediction = xi.dot(theta)
                error = prediction - yi
                gradient = xi.T.dot(error)
                theta -= learning_rate * gradient
                cost = (1/(2*m)) * np.sum(error**2)
                cost_history.append(cost)
        return theta
    
    def mini_batch_gradient_descent(self,X, y, batch_size, learning_rate, epochs):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        
        for _ in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, m, batch_size):
                xi = X_shuffled[i:i+batch_size]
                yi = y_shuffled[i:i+batch_size]
                predictions = xi.dot(theta)
                errors = predictions - yi
                gradient = (1/batch_size) * xi.T.dot(errors)
                theta -= learning_rate * gradient
                
                cost = (1/(2*batch_size)) * np.sum(errors**2)
                cost_history.append(cost)
                
        return theta

    def train_lr(self):
        try:
            self.result_path = ''
            file_path = self.dataset_name.toPlainText()
            Dataset_name = file_path.split('/')[-1]
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.txt'):
                data = np.loadtxt(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                raise Exception('Unsupported file type')

            X = data.iloc[1:,:-1].values
            y = data.iloc[1:,-1].values

            test_size = float(self.gradient_descent_testsize.value())

            #划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

            #选择梯度下降方法
            method = self.method.currentText()
            learning_rate = float(self.learning_rate.value())
            epochs = int(self.epochs.value())
            batch_size = int(self.batch_size.value())
            if method == '批梯度下降':
                theta = self.batch_gradient_descent(X_train, y_train, learning_rate, epochs)
            elif method == '随机梯度下降':
                theta = self.stochastic_gradient_descent(X_train, y_train, learning_rate, epochs)
            elif method == '小批量梯度下降':
                theta = self.mini_batch_gradient_descent(X_train, y_train, batch_size, learning_rate, epochs)
            else:
                raise ValueError("Invalid method specified")
            
            # 预测
            y_pred = X_test.dot(theta)
            #计算均方误差
            mse = np.mean((y_test - y_pred) ** 2)

            #显示结果
            result_text = f"Dataset：{Dataset_name}\nmethod: {method}\nlearning_rate:{learning_rate}\nepochs: {epochs}\nbatch_size: {batch_size}\nMean Squared Error: {mse:.4f}\nPredictions: {y_pred}"
            self.textBrowser.setText(result_text)
            if self.selected_group:
                self.group_states[self.selected_group]['textBrowser'] = result_text

            # 结果可视化
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title('True vs Predicted Values')
            #生成随机文件名
            random_name = ''.join(random.choices(string.ascii_letters + string.digits,k=10))+'.jpg'
            result_path = os.path.join('results',random_name)
            #确保结果文件夹存在
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            plt.savefig(result_path)
            self.result_img_path.setText(f"Image Path: {result_path}")
            self.lbl_img.setPixmap(QPixmap(result_path))
            if self.selected_group:
                self.page_states[self.selected_button]['textBrowser'] = result_text
                self.page_states[self.selected_button]['image_path'] = result_path
                self.page_states[self.selected_button]['lbl_img'] = 'result_path'
                self.page_states[self.selected_button]['dataset'] = Dataset_name
                self.page_states[self.selected_button]['gradient_descent_testsize'] = self.gradient_descent_testsize.value()
                self.page_states[self.selected_button]['method'] = self.method.currentText()
                self.page_states[self.selected_button]['learning_rate'] = self.learning_rate.value()
                self.page_states[self.selected_button]['epochs'] = self.epochs.value()
                self.page_states[self.selected_button]['batch_size'] = self.batch_size.value()
            #保存当前页面状态
            self.result_path = result_path
            self.save_page_state(self.selected_button)
        except Exception as e:
            print(f"Error in train_lr: {e}")

    def show_context_menu(self,pos):
        sender = self.sender()
        if sender.text() == '[Blank Page]':
            #创建右键菜单
            menu = QMenu(self)
            action1 = QAction("添加页面",self)
            menu.addAction(action1)

            #连接动作到相应的槽函数
            if sender:
                action1.triggered.connect(partial(self.action1_triggered, sender))  # 使用 partial 绑定参数
            else:
                print("show_context_menu called with button: None")  # 调试信息

            #显示菜单
            menu.exec_(self.sender().mapToGlobal(pos))
        else:
            #创建右键菜单
            menu = QMenu(self)
            action2 = QAction("删除页面",self)
            menu.addAction(action2)
            action3 = QAction("修改页面名称",self)
            menu.addAction(action3)

            #连接动作到相应的槽函数
            if sender:
                action2.triggered.connect(partial(self.action2_triggered, sender))  # 使用 partial 绑定参数
                action3.triggered.connect(partial(self.action3_triggered, sender))
            else:
                print("show_context_menu called with button: None")  # 调试信息

            #显示菜单
            menu.exec_(self.sender().mapToGlobal(pos))
    
    def action1_triggered(self,button):
        dialog = QDialog()
        ui = Ui_Dialog()
        ui.setupUi(dialog)

        # 连接OK按钮的点击事件
        ui.pushButton.clicked.connect(lambda:self.update_button_text(button,ui.lineEdit.text(),dialog))
        ui.pushButton_2.clicked.connect(dialog.reject)#关闭对话框
        #显示对话框
        dialog.exec()
    
    def action2_triggered(self,button):
        button.setText('[Blank Page]')
        self.remove_button_from_selected_group(button)
        self.page_states.pop(button, None) #移除页面状态
        button.setStyleSheet("")
        self.textBrowser.clear()
        self.lbl_img.setPixmap(QPixmap(""))
        self.dataset_name.clear()
        self.gradient_descent_testsize.setValue(0.1)
        self.method.setCurrentText('批梯度下降')
        self.learning_rate.setValue(0.01)
        self.epochs.setValue(100)
        self.batch_size.setValue(32)
        self.result_img_path.clear()
    
    def action3_triggered(self,button):
        dialog = QDialog()
        ui = Ui_Dialog()
        ui.setupUi(dialog)
        # 连接OK按钮的点击事件
        ui.pushButton.clicked.connect(lambda:self.update_button_text_action3(button,ui.lineEdit.text(),dialog))
        ui.pushButton_2.clicked.connect(dialog.reject)#关闭对话框
        #显示对话框
        dialog.exec()
    
    def update_button_text_action3(self,button,text,dialog):
        button.setText(text)
        if text != '[Blank Page]':
            self.groups[self.selected_group].append(button)
            button.setStyleSheet("background-color:yellow;")
            #保存新添加页面的状态
            #self.save_page_state(button)
            # 自动选择新添加页面
            # 恢复之前选中的按钮的样式
        if self.selected_button and self.selected_button != button and self.selected_button != self.selected_group:
            self.selected_button.setStyleSheet("")
        dialog.reject()

    def update_button_text(self,button,text,dialog):
        try:
            button.setText(text)
            if text != '[Blank Page]':
                self.add_button_to_selected_group(button)
                #保存新添加页面的状态
                #self.save_page_state(button)
                # 自动选择新添加页面
                # 恢复之前选中的按钮的样式
            if self.selected_button and self.selected_button != button and self.selected_button != self.selected_group:
                self.selected_button.setStyleSheet("")
                self.save_page_state(self.selected_button)
            #更新当前选中的按钮
            self.selected_button = button
            #清空文本框和图片
            button.setStyleSheet("background-color:yellow;")
            self.textBrowser.clear()
            self.lbl_img.setPixmap(QPixmap(""))
            self.dataset_name.clear()
            self.gradient_descent_testsize.setValue(0.1)
            self.method.setCurrentText('批梯度下降')
            self.learning_rate.setValue(0.01)
            self.epochs.setValue(100)
            self.batch_size.setValue(32)
            self.result_img_path.clear()
            #更新最后一次使用的按钮
            if self.selected_group:
                self.group_states[self.selected_group]['last_used_button'] = button
            dialog.accept()  # 接受对话框，关闭对话框
        except Exception as e:
            print(f"Error in update_button_text: {e}")
    def add_button_to_selected_group(self,button):
        if self.selected_group:
            self.groups[self.selected_group].append(button)
            print(f"Added {button.text()} to group {self.selected_group.text()}")
            button.setStyleSheet("background-color:yellow;")
            self.page_states[button] = {'textBrowser':'',
                                        'lbl_img':'',
                                        'dataset':'',
                                        'gradient_descent_testsize':0.1,
                                        'method':'批梯度下降',
                                        'learning_rate':0.01,
                                        'epochs':100,
                                        'batch_size':32,
                                        'image_path':''}
            #保存新添加页面的状态
            #self.save_page_state(button)
            #自动选择新添加的页面
            #self.select_page(button)
            #刷新页面
            #self.restore_page_state(button)
    def remove_button_from_selected_group(self, button):
        for group_buttons in self.groups.values():
            if button in group_buttons:
                group_buttons.remove(button)
                print(f"Removed {button.text()} from group {self.selected_group.text()}")
                button.setStyleSheet("") # 恢复默认背景颜色
                #移除页面的状态
                self.page_states.pop(button, None)
                #如果移除的是当前选中的页面， 重置选中的页面
                if self.selected_button == button:
                    self.selected_button = None

    def save_group_state(self, group_button):
        print("save group state")
        if group_button is None:
            group_button = self.selected_group
        if group_button:
            file_path = self.result_img_path.toPlainText()
            image_path_name = file_path.split(' ')[-1]
            state = {'buttons':[], 
                     'textBrowser': self.textBrowser.toPlainText(),
                     'lbl_img': image_path_name,
                     'dataset':self.dataset_name.toPlainText(),
                     'gradient_descent_testsize':self.gradient_descent_testsize.value(),
                     'method':self.method.currentText(),
                     'learning_rate':self.learning_rate.value(),
                     'epochs':self.epochs.value(),
                     'batch_size':self.batch_size.value(),
                     'image_path':self.result_img_path.toPlainText()
                     #'last_used_button': self.selected_button #保存最后一次使用的按钮
            }
        if group_button != self.selected_button:
            state['last_used_button'] = self.selected_button
            for btn in self.buttons:
                state['buttons'].append((btn,btn.text(),btn.isEnabled()))
            self.group_states[group_button] = state
            print(f"Saved state for group {group_button.text()}: {state}")
            print(f"Group button object: {group_button}")  # 添加调试信息
        else:
            print("No group button to save state")

    def save_page_state(self,button):
        try:
            if button:
                file_path = self.result_img_path.toPlainText()
                image_path_name = file_path.split(' ')[-1]
                state = {
                    'textBrowser': self.textBrowser.toPlainText(),
                    'lbl_img': image_path_name,
                    'dataset': self.dataset_name.toPlainText(),
                    'gradient_descent_testsize': self.gradient_descent_testsize.value(),
                    'method': self.method.currentText(),
                    'learning_rate': self.learning_rate.value(),
                    'epochs': self.epochs.value(),
                    'batch_size': self.batch_size.value(),
                    'image_path': self.result_img_path.toPlainText()
                }
                self.page_states[button] = state
                print(f"Saved state for page {button.text()}: {state}")
            else:
                print("No button to save state")
        except Exception as e:
            print(f"Error in save_page_state: {e}")

    def restore_group_state(self, group):
        print("now restore group state")
        if group in self.group_states:
            file_path = self.result_img_path.toPlainText()
            image_path_name = file_path.split(' ')[-1]
            state = self.group_states[group]
            for btn, text, enabled in state['buttons']:
                btn.setText(text)
                btn.setEnabled(enabled)
                '''if btn in self.groups[group] and btn == group:
                    btn.setStyleSheet("background-color:yellow;")
                else: 
                    btn.setStyleSheet("")'''
            self.textBrowser.setText(state['textBrowser'])
            self.lbl_img.setPixmap(QPixmap(state['lbl_img']))
            self.dataset_name.setText(state['dataset'])
            self.gradient_descent_testsize.setValue(state['gradient_descent_testsize'])
            if 'image_path' in state:  # 确保 image_path 存在
                self.result_img_path.setText(state['image_path'])
            print(f"Restored state for group {group.text()}: {state}")
            print(f"Group button object: {group}")  # 添加调试信息
        else:
            print(f"No saved state for group {group.text()}, resetting all buttons to [Blank Page]")
            for btn in self.groups[group]:
                if btn.text() != '[Blank Page]':
                    btn.setText('[Blank Page]')
                    btn.setEnabled(True)
                    btn.setStyleSheet("")
            self.textBrowser.clear()
            self.lbl_img.setPixmap(QPixmap(""))
            self.dataset_name.clear()
            self.gradient_descent_testsize.setValue(0.1)
            self.method.setcurrentText("批梯度下降")
            self.learning_rate.setValue(0.01)
            self.epochs.setValue(100)
            self.batch_size.setValue(32)
            self.result_img_path.clear()
            print(f"No saved state for group {group.text()}, resetting all buttons to [Blank Page]")

    def select_group(self, button):
        try:
            # 保存当前选中的组别的状态
            self.save_group_state(self.selected_group)
            #重置之前选中的组和按钮
            if self.selected_group:
                self.selected_group.setStyleSheet("")#恢复默认样式
            self.selected_group = button
            print(f"Selected group: {self.selected_group.text()}")
            self.selected_group.setStyleSheet("background-color:yellow;")#设置选中的组别为黄色
            #恢复当前选中的组别的状态
            self.restore_group_state(button)
            #重置所有按钮的样式
            for btn in self.buttons:
                btn.setEnabled(True)
            # 重置当前选中的按钮
            self.selected_button = None
            #自动选择该组别最后一次选择使用的页面
            last_used_button = self.group_states[button].get('last_used_button')
            print(f"Last used button: {last_used_button}")
            if last_used_button:
                print(f"Last used button text: {last_used_button.text()}")
            if last_used_button and last_used_button.text() != '[Blank Page]':
                self.select_page(last_used_button)
            else:
                #如果没有找到，则将所有页面的背景颜色设置为默认颜色
                for btn in self.page_buttons:
                    btn = getattr(self, btn, None)
                    btn.setStyleSheet("")
                print(f"No last used button found for group {button.text()}")
            # 立即调用change_button_color方法更新按钮的颜色
            #self.change_button_color(button)
        except Exception as e:
            print(f"Error in select_group: {e}")

    def select_page(self, button):
        print("now select page")
        # 恢复之前选中的按钮的样式
        if self.selected_button and self.selected_button != button and self.selected_button != self.selected_group:
            self.selected_button.setStyleSheet("")
        #更新当前选中的按钮
        self.selected_button = button
        button.setStyleSheet("background-color:yellow;")
        print(f"Selected page: {button.text()}")
        #保存之前选中的页面状态
        self.save_page_state(self.group_states[self.selected_group]['last_used_button'])
        # 恢复页面状态
        self.restore_page_state(button)
        #更新最后一次使用的按钮
        if self.selected_group:
            self.group_states[self.selected_group]['last_used_button'] = button
    def on_button_pressed(self):
        try:
            # 获取发送信号的对象（即被按下的按钮）
            button = self.sender()
            if button and button.text() != '[Blank Page]' and button.isEnabled():
                self.change_button_color(button)
        except Exception as e:
            print(f"Error in on_button_pressed: {e}")
    
    def on_button_released(self):
        try:
            # 获取发送信号的对象（即被释放的按钮）
            button = self.sender()
            if button is None or button.text() == '[Blank Page]' or not button.isEnabled():
                return  # 如果按钮对象无效，直接返回
            # 恢复之前选中的按钮的样式
            if self.selected_button and self.selected_button != button:
                self.selected_button.setStyleSheet("")
            # 更新当前选中的按钮
            self.selected_button = button
            #self.restore_page_state(button)
        except Exception as e:
            print(f"Error in on_button_released: {e}")
    
    def change_button_color(self, button):
        try:
            if button:
                print(f"change_button_color: button text: {button.text()}")
                if self.selected_group:
                    print(f"change_button_color: selected group: {self.selected_group.text()}")
                    # 恢复之前选中的按钮的样式
                    for btn in self.groups[self.selected_group]:
                        if btn != button and btn.text() != '[Blank Page]':
                            btn.setStyleSheet("")
                    # 更新当前选中的按钮
                    self.selected_button = button
                    button.setStyleSheet("background-color: yellow;")
                    #更新最后一次使用的页面按钮
                    if self.selected_group.objectName() == button.objectName():
                        self.group_states[self.selected_group]['last_used_button'].setStyleSheet("background-color: yellow;")
                    print(f"Selected button: {button.text()}")
                    self.select_page(button)
                else:
                    print("change_button_color: selected_group is None")
            else:
                print("change_button_color: button is None")
        except Exception as e:
            print(f"Error in change_button_color: {e}")
    
    def restore_page_state(self, button):
        if button in self.page_states:
            state = self.page_states[button]
            self.textBrowser.setText(state['textBrowser'])
            self.lbl_img.setPixmap(QPixmap(state['lbl_img']))
            self.dataset_name.setText(state['dataset'])
            self.gradient_descent_testsize.setValue(state['gradient_descent_testsize'])
            self.method.setCurrentText(state['method'])
            self.learning_rate.setValue(state['learning_rate'])
            self.epochs.setValue(state['epochs'])
            self.batch_size.setValue(state['batch_size'])
            self.result_img_path.setText(state['image_path'])
            print(f"Restored state for button {button.text()}: {state}")
        else:
            print(f"No saved state for button {button.text()}, resetting all buttons to [Blank Page]")
            self.textBrowser.clear()
            self.lbl_img.setPixmap(QPixmap(""))
            self.dataset_name.clear()
            self.gradient_descent_testsize.setValue(0.1)
            self.method.setCurrentText("批梯度下降")
            self.learning_rate.setValue(0.01)
            self.epochs.setValue(100)
            self.batch_size.setValue(32)
            self.result_img_path.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Gradient_Descent_Window()
    win.show()
    sys.exit(app.exec_())