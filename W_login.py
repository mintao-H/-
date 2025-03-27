from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from Ui_login import Ui_MainWindow
import sys
from W_main import MainWindow


class LoginWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(LoginWindow, self).__init__()
        self.setupUi(self)
        self.btn_submit.clicked.connect(self.logining)
        self.btn_exit.clicked.connect(self.close)
    
    def logining(self):
        user = self.txt_user.text()
        pwd = self.txt_password.text()
        if user == "admin" and pwd == "admin":
            # print("success")
            self.hide()
            self.win_m = MainWindow()
            self.win_m.show()
        else:
            QMessageBox.warning(self, "警告", "用户名或密码错误！")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoginWindow()
    win.show()
    sys.exit(app.exec_())