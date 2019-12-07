from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pydot
# import pygraphviz as pgv
from IPython.display import Image


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
		testword = "obama"

		self.label_history.setText(self.text_history)

		# if self.TF_IDF.isClicked():
		# 	mode = "TF-IDF"
		if self.SG.isChecked():
			mode = "SG"
		elif self.CBOW.isChecked():
			mode = "CBOW"
		else:
			print("wrong mode")
		# W = [W_emb, W_out]
		# testwords = ["Obama", "senator", "white", "house", "battle"]
		# for tw in testwords:
		#     sim(tw,word2ind,ind2word,W_emb)
		# mode = "SG"
		parameters = open(mode+'_parameters.pickle','r')
		W_emb = parameters[0]
		word2ind = parameters[2]
		# testwords = ["Obama", "petition", "regard", "one", "housing", "owner"]
		# for testword in testwords:

		length = (W_emb*W_emb).sum(1)**0.5
		wi = word2ind[text_data]
		inputVector = W_emb[wi].reshape(1,-1)/length[wi]
		sim = (inputVector@matrix.t())[0]/length
		values, indices = sim.squeeze().topk(keyword_number)
		G = pgv.AGraph()
		# centernode = testword
		# nodelist=["white", "america", "white house", "eminem"]

		
		print()
		print("===============================================")
		print("The most similar words to \"" + testword + "\"")
		for ind, val in zip(indices,values):
			nodelist.append(ind2word[ind.item()])
		print(ind2word[ind.item()]+":%.3f"%(val,))
		print("===============================================")
		print()


		G.add_nodes_from(nodelist)
		G.add_node(centernode) # adds center node

		# Setting node attributes that are common for all nodes 
		G.node_attr['style']='filled'
		G.node_attr['color']='#FFFFFF'
		G.node_attr['fontcolor']= 'red'
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
		center.attr['color']='lightgray'
		center.attr['fontsize']='50'
		center.attr['fontcolor']='black'

		for i in nodelist:
			G.add_edge(centernode,i)
			n = G.get_node(i)
			n.attr['fontcolor']="#%2x0000"%(sim_dict[i]) # random?
			n.attr['height']="%s"%(sim_dict[i]*0.3)
			n.attr['fontsize']="%s"%(sim_dict[i]*8 + 30) # 관련성에 비례하게

		G.draw('viz.png',prog="circo") # This creates a .png file in the local directory. Displayed below.

		Image('viz.png', width=700) # The Graph visualization we created above.
	
		self.label.setParent(None)
		self.pixmap = QPixmap('viz.png')
		self.label.resize(700, 500)
		self.pixmap = self.pixmap.scaled(700, 500)
		self.label.setPixmap(self.pixmap)

		self.resultLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)


		# G = pgv.AGraph()
		# centernode = 'chogook'
		# nodelist=['chomin','president moon','Trump','controversy','law','professor','politics']
		# G.add_nodes_from(nodelist)
		# G.add_node(centernode) # adds center node

		# # Setting node attributes that are common for all nodes 
		# G.node_attr['style']='filled'
		# G.node_attr['color']='#FFFFFF'
		# G.node_attr['fontcolor']= 'red'
		# G.edge_attr['style'] = 'invis'

		# # similarity dictionary
		# sim_dict = {'chomin':9,'president moon':5,'Trump':2,'controversy':8,'law':1,'professor':10,'politics':7}

		# # Creating and setting node attributes that vary for each node (using a for loop)
		# center = G.get_node(centernode)
		# center.attr['style']='filled'
		# center.attr['shape']='circle'
		# center.attr['color']='lightgray'
		# center.attr['fontsize']='50'
		# center.attr['fontcolor']='black'

		# for i in nodelist:
		#     G.add_edge(centernode,i)
		#     n = G.get_node(i)
		#     n.attr['fontcolor']="#%2x0000"%(sim_dict[i]) # random?
		#     n.attr['height']="%s"%(sim_dict[i]*0.3)
		#     n.attr['fontsize']="%s"%(sim_dict[i]*8 + 30) # 관련성에 비례하게

		# G.draw('viz.png',prog="circo") # This creates a .png file in the local directory. Displayed below.

		# Image('viz.png', width=700) # The Graph visualization we created above.

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
