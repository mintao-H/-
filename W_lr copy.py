from PyQt5.QtWidgets import QVBoxLayout,QApplication, QGraphicsPixmapItem,QMainWindow, QMessageBox,QMenu,QAction,QFileDialog,QDialog,QGraphicsView, QGraphicsScene
from Ui_logistic_regression import Ui_MainWindow
from PyQt5.QtGui import QPixmap,QPainter
from PyQt5.QtCore import Qt
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ui_add_page import Ui_Dialog
from functools import partial
from Ui_Image_Viewer import Ui_Image_show
import os
import random
import string
class LRWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(LRWindow, self).__init__()
        self.setupUi(self)
        pixmap = QPixmap(None)
        self.result_path = ''
        self.dataset_name.setText(None)
        self.lbl_img.setPixmap(pixmap)
        self.lbl_img.setScaledContents(True)#设置图像自动缩放
        self.btn_train.clicked.connect(self.train_lr)
        # 连接 lbl_img 的点击事件
        self.lbl_img.mousePressEvent = self.show_image_popup
        self.DSB_lr.setValue(0.01)
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
                self.group_states[button] = {'buttons':[],'textBrowser':'','lbl_img':'','dataset':'','learning_rate':0.01,'image_path':'','last_used_button': None} #初始化组别状态
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
                self.page_states[button] = {'buttons':[],'textBrowser':'','lbl_img':'','dataset':'','learning_rate':0.01,'image_path':''}

        #初始化每个组别的页面按钮状态
        for group_button in self.groups.keys():
            state = {'buttons':[], 
                     'textBrowser': self.textBrowser.toPlainText(),
                     'lbl_img':'',
                     'dataset':'',
                     'learning_rate':0.01,
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
        #创建一个QGraphicsView 和 QGraphicsScene 来显示图像
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        layout = QVBoxLayout(dialog)
        layout.addWidget(view)
        #应用鼠标滚轮缩放
        view.setRenderHint(QPainter.Antialiasing)
        view.setDragMode(QGraphicsView.ScrollHandDrag)
        view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
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

            #划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

            # 创建并训练逻辑回归模型
            model = LogisticRegression()
            model.fit(X_train, y_train)

            #预测
            y_pred = model.predict(X_test)

            #计算准确率
            accuracy = accuracy_score(y_test, y_pred)

            #显示结果
            result_text = f"Dataset：{Dataset_name}\nAccuracy: {accuracy:.4f}\nPredictions: {y_pred}"
            self.textBrowser.setText(result_text)
            if self.selected_group:
                self.group_states[self.selected_group]['textBrowser'] = result_text

            #结果可视化
            axs,fig = plt.subplots(1,2)
            fig[0].scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis',marker = 'o')
            fig[0].set_title('Predicted')
            fig[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis',marker = 'o')
            fig[1].set_title('Actual')
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
                self.page_states[self.selected_button]['learning_rate'] = self.DSB_lr.value()
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
        self.DSB_lr.setValue(0.01)
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
            self.DSB_lr.setValue(0.01)
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
            self.page_states[button] = {'textBrowser':'','lbl_img':'','dataset':'','learning_rate':0.01,'image_path':''}
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
                     'learning_rate':self.DSB_lr.value(),
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
                    'learning_rate': self.DSB_lr.value(),
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
            self.DSB_lr.setValue(state['learning_rate'])
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
            self.DSB_lr.setValue(0.01)
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
            self.DSB_lr.setValue(state['learning_rate'])
            self.result_img_path.setText(state['image_path'])
            print(f"Restored state for button {button.text()}: {state}")
        else:
            print(f"No saved state for button {button.text()}, resetting all buttons to [Blank Page]")
            self.textBrowser.clear()
            self.lbl_img.setPixmap(QPixmap(""))
            self.dataset_name.clear()
            self.DSB_lr.setValue(0.01)
            self.result_img_path.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LRWindow()
    win.show()
    sys.exit(app.exec_())