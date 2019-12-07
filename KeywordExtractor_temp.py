# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KeywordExtractor1.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1025, 706)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(220, 170, 811, 531))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.resultLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.resultLayout.setContentsMargins(0, 0, 0, 0)
        self.resultLayout.setObjectName("resultLayout")
        self.SG = QtWidgets.QRadioButton(Form)
        self.SG.setGeometry(QtCore.QRect(20, 140, 121, 51))
        self.SG.setObjectName("SG")
        self.CBOW = QtWidgets.QRadioButton(Form)
        self.CBOW.setGeometry(QtCore.QRect(20, 180, 121, 51))
        self.CBOW.setObjectName("CBOW")
        self.RNN = QtWidgets.QRadioButton(Form)
        self.RNN.setGeometry(QtCore.QRect(20, 220, 121, 51))
        self.RNN.setObjectName("RNN")
        self.confirm_button = QtWidgets.QPushButton(Form)
        self.confirm_button.setGeometry(QtCore.QRect(740, 40, 93, 28))
        self.confirm_button.setObjectName("confirm_button")
        self.search_line = QtWidgets.QLineEdit(Form)
        self.search_line.setGeometry(QtCore.QRect(370, 30, 331, 61))
        self.search_line.setObjectName("search_line")
        self.model_option_button = QtWidgets.QToolButton(Form)
        self.model_option_button.setGeometry(QtCore.QRect(10, 120, 121, 21))
        self.model_option_button.setObjectName("model_option_button")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 590, 121, 111))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.reset_button = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.reset_button.setObjectName("reset_button")
        self.verticalLayout_4.addWidget(self.reset_button)
        self.exit_button = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.exit_button.setObjectName("exit_button")
        self.verticalLayout_4.addWidget(self.exit_button)
        self.model_option_button_2 = QtWidgets.QToolButton(Form)
        self.model_option_button_2.setGeometry(QtCore.QRect(10, 330, 121, 21))
        self.model_option_button_2.setObjectName("model_option_button_2")
        self.search_line_2 = QtWidgets.QLineEdit(Form)
        self.search_line_2.setGeometry(QtCore.QRect(10, 360, 111, 31))
        self.search_line_2.setObjectName("search_line_2")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(220, 130, 340, 41))
        self.label.setObjectName("label")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.SG.setText(_translate("Form", "Skip-Gram"))
        self.CBOW.setText(_translate("Form", "CBOW"))
        self.RNN.setText(_translate("Form", "RNN"))
        self.confirm_button.setText(_translate("Form", "Confirm"))
        self.model_option_button.setText(_translate("Form", "MODEL"))
        self.reset_button.setText(_translate("Form", "Reset"))
        self.exit_button.setText(_translate("Form", "Exit"))
        self.model_option_button_2.setText(_translate("Form", "Keywords"))
        self.label.setText(_translate("Form", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
