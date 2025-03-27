from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QMessageBox
# from PyQt5.QtWidgets import QMainWindow
from Ui_cal import Ui_MainWindow
import sys
import requests


class cal_Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(cal_Window, self).__init__()
        self.setupUi(self)

        self.btn_cal.clicked.connect(self.cal)
        self.btn_exit.clicked.connect(self.close)
        self.btn_update.clicked.connect(self.show_weather)
        self.select_city.currentIndexChanged.connect(self.show_weather)
    
    def show_weather(self):
        api_key = 'SG5M4LUyFYidnSjYR'
        city = self.select_city.currentText()
        response = requests.get(f'https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={city}&language=zh-Hans&unit=c')
        result = response.json()
        txt_city = result['results'][0]['location']['name']
        txt_wea = result['results'][0]['now']['text']
        txt_tem = result['results'][0]['now']['temperature']
        self.lbl_city.setText("城市："+txt_city)
        self.lbl_wea.setText("天气："+txt_wea)
        self.lbl_tem.setText("温度："+txt_tem + "℃")
        

    def cal(self):
        try:
            a = self.txt_data1.toPlainText()
            b = self.txt_data2.toPlainText()
            a = float(a)
            b = float(b)
            method = self.select_method.currentText()
            if method == "＋":
                c = a + b
            elif method == "－":
                c = a - b
            elif method == "×":
                c = a * b
            else:
                if b == 0:
                    QMessageBox.warning(self, "警告", "除数不能为0！!!")
                else:
                    c = a / b
                    c = str(c)
                    self.txt_result.setText(c)
        except ValueError:
            # print("请输入数字！")
            QMessageBox.warning(self, "警告", "请输入数字！!!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = cal_Window()
    win.show()
    sys.exit(app.exec_())