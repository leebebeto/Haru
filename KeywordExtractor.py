from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pydot
import pygraphviz as pgv
from IPython.display import Image
import pickle
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
import os
import re
import pandas as pd 
import numpy as np
import math
import operator
import nltk

class Ui_Form(object):
	def setupUi(self, Form):
		Form.setObjectName("Form")
		Form.resize(1025, 706)
		self.verticalLayoutWidget = QtWidgets.QWidget(Form)
		self.verticalLayoutWidget.setGeometry(QtCore.QRect(220, 110, 811, 591))
		self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
		self.resultLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
		self.resultLayout.setContentsMargins(0, 0, 0, 0)
		self.resultLayout.setObjectName("resultLayout")
		self.TF_IDF = QtWidgets.QRadioButton(Form)
		self.TF_IDF.setGeometry(QtCore.QRect(20, 140, 121, 51))
		self.TF_IDF.setObjectName("TF_IDF")
		self.SG = QtWidgets.QRadioButton(Form)
		self.SG.setGeometry(QtCore.QRect(20, 180, 121, 51))
		self.SG.setObjectName("SG")
		self.CBOW = QtWidgets.QRadioButton(Form)
		self.CBOW.setGeometry(QtCore.QRect(20, 220, 121, 51))
		self.CBOW.setObjectName("CBOW")
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
		self.keyword_option = QtWidgets.QToolButton(Form)
		self.keyword_option.setGeometry(QtCore.QRect(10, 330, 121, 21))
		self.keyword_option.setObjectName("keyword_option")
		self.keyword_num = QtWidgets.QLineEdit(Form)
		self.keyword_num.setGeometry(QtCore.QRect(10, 360, 111, 31))
		self.keyword_num.setObjectName("keyword_num")

		self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
		self.label.setMaximumSize(QtCore.QSize(980, 600))
		self.label.setObjectName("label")

		self.label_history = QtWidgets.QLabel(Form)
		self.label_history.setGeometry(QtCore.QRect(220, 100, 340, 41))
		self.label_history.setObjectName("label_history")


		self.exit_button.clicked.connect(self.exit_button_clicked)
		self.reset_button.clicked.connect(self.reset_button_clicked)
		self.confirm_button.clicked.connect(self.confirm_button_clicked)


		self.text_history = ""

		self.retranslateUi(Form)
		QtCore.QMetaObject.connectSlotsByName(Form)


	def confirm_button_clicked(self):
		text_data = self.search_line.text()
		if self.text_history == "":	
			print("1 self.text_history", self.text_history)		
			self.text_history += text_data
			print("2 self.text_history", self.text_history)		
		else:
			self.text_history += " -> " + text_data
		keyword_number = self.keyword_num.text()
		
		print(text_data)
		print(keyword_number)
		print(self.SG.isChecked())
		print(self.CBOW.isChecked())
		print(self.TF_IDF.isChecked())
		print(self.text_history)

		self.label_history.setText(self.text_history)
		nodelist = []

		if self.SG.isChecked() or self.CBOW.isChecked():
			if self.SG.isChecked():
				mode = "SG"
			elif self.CBOW.isChecked():
				mode = "CBOW"
			else:
				print("wrong mode")

			with open(mode+'_parameters.pickle','rb') as f:
				parameters = pickle.load(f)
			W_emb = parameters[0]
			word2ind = parameters[2]
			ind2word = parameters[3]

			centernode = text_data
			length = (W_emb*W_emb).sum(1)**0.5
			wi = word2ind[text_data]
			inputVector = W_emb[wi].reshape(1,-1)/length[wi]
			sim = (inputVector@W_emb.t())[0]/length
			values, indices = sim.squeeze().topk(80)

			iteration = 0
			for ind, val in zip(indices,values):
				if len(ind2word[ind.item()]) > 5:
					nodelist.append(ind2word[ind.item()])
					iteration += 1
				if iteration == int(keyword_number):
					break

		elif self.TF_IDF.isChecked():
			file_root = os.path.dirname(os.path.abspath(__file__))
			with open(file_root+'\\query.txt', 'w') as f:
				f.write(text_data)

			with open(file_root+'\\query.txt', 'r') as f:
				query = [line.strip() for line in f][0]

			with open(file_root+'\\tf_idf_parameters.pickle', 'rb') as f:
				saved_data = pickle.load(f)
			os.system('python tf_idf_1.py')			

			with open(file_root+'\\final_result.pickle', 'rb') as f:
				final_result = pickle.load(f)

			final_result = final_result[0:int(keyword_number)]
			for item in final_result:
				nodelist.append(item[:-4])
			print('its nodelist')
			print(nodelist)

		G = pgv.AGraph(bgcolor="#EEEEEE")

		G.add_nodes_from(nodelist)
		G.add_node(centernode) # adds center node

		# Setting node attributes that are common for all nodes
		G.node_attr['style']='filled'
		G.node_attr['color']='#FFFFFF'
		G.node_attr['fontcolor']= 'black'
		G.edge_attr['style'] = 'invis'

		# similarity dictionary
		sim_dict = {}
		for i, item in enumerate(nodelist):
			sim_dict[item] = 2*len(nodelist)-2*i
		print(sim_dict)
		# Creating and setting node attributes that vary for each node (using a for loop)
		center = G.get_node(centernode)
		center.attr['style']='filled'
		center.attr['shape']='circle'
		center.attr['color']='transparent'
		center.attr['fontsize']='50'
		center.attr['fontcolor']='black'
		center.attr['width']='4'
		center.attr['height']='4'

		
		for i in nodelist:
			G.add_edge(center,i)
			n = G.get_node(i)
			n.attr['fontcolor']="#%1x0000"%(sim_dict[i]) # random?
			n.attr['height']="%s"%(sim_dict[i]*0.3)
			n.attr['fontsize']="%s"%(sim_dict[i]*20 + 100) # 관련성에 비례하게
			n.attr['color'] = 'transparent'
		G.draw('viz.png',prog="circo") # This creates a .png file in the local directory. Displayed below.

		Image('viz.png', width=700) # The Graph visualization we created above.

	def reset_button_clicked(self):
		self.search_line.setText("")
		self.keyword_num.setText("")
		self.text_history = ""
		self.label_history.setText("")
		

		self.label.setParent(None)

	def exit_button_clicked(self):
		sys.exit(QtWidgets.QApplication(sys.argv).exec_())


	def retranslateUi(self, Form):
		_translate = QtCore.QCoreApplication.translate
		Form.setWindowTitle(_translate("Form", "Form"))
		self.SG.setText(_translate("Form", "Skip-Gram"))
		self.CBOW.setText(_translate("Form", "CBOW"))
		self.TF_IDF.setText(_translate("Form", "TF_IDF"))
		self.confirm_button.setText(_translate("Form", "Confirm"))
		self.model_option_button.setText(_translate("Form", "MODEL"))
		self.reset_button.setText(_translate("Form", "Reset"))
		self.exit_button.setText(_translate("Form", "Exit"))
		self.keyword_option.setText("Keyword")


if __name__ == "__main__":
	import sys
	app = QtWidgets.QApplication(sys.argv)
	Form = QtWidgets.QWidget()
	ui = Ui_Form()
	ui.setupUi(Form)
	Form.show()
	sys.exit(app.exec_())
