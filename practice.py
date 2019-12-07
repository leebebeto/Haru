# import os 
# import re
# import pickle
# file_list = os.listdir(os.getcwd()+'/cnn_news')
# text_total_data= []
# text = ""
# for i in range(2000):
# 	if i %1000 == 0:
# 		print(i)
# 	file_path = os.getcwd()+'/cnn_news/'+file_list[i]
# 	temp_result = [] 	
# 	with open(file_path,'r', encoding= 'utf-8') as f:
# 		data = f.readlines()
# 		for line in data:
# 			line = line.replace("\n", "")
# 			line = line.replace("@highlight", "")
# 			temp_result.append(line)
# 	text_total_data.append(temp_result)
# for lines in text_total_data:
# 	text += str(lines[1:-1]) 
# # text_total_data = str(text_total_data[0])
# # print(len(text_total_data))
# with open('text_total_data.txt', 'w', encoding = 'utf-8') as f:
# 	f.write(text)
# print(len(text))
# # import pickle
# # with open('SG_parameters.pickle', 'rb') as f:
# #   input_data = pickle.load(f)
# # print(input_data)
# # data = open('text_total_data.pickle', 'r').read() # should be simple plain text file
# # data = ""
# # for lists in input_data[0:2]:
# #   for word in lists:
# #     data += word
# # data = data.split(' ')
# # chars = list(set(data))
# # print(chars)


import os 
import re
import pickle
file_list = os.listdir(os.getcwd()+'/cnn_news')
text_total_data= []
text = ""
for i in range(2000):
	if i %1000 == 0:
		print(i)
	file_path = os.getcwd()+'/cnn_news/'+file_list[i]
	temp_result = [] 	
	with open(file_path,'r', encoding= 'utf-8') as f:
		title = ""
		data = f.readlines()
		print('123123', str(data[0]))
		temp = data[0].split(' ')
		temp = temp[:10]
		for word in temp:
			title += word + ' ' 
		print(title)
		for line in data:
			line = line.replace("\n", "")
			line = line.replace("@highlight", "")
			temp_result.append(line) 

		try:
			f = open(os.getcwd()+'\\data_txt\\'+title+'.txt','w')
			f.write(str(temp_result[1:-1]))
		except:
			print(i)
# 	text_total_data.append(temp_result)
# for lines in text_total_data:
# 	text += str(lines[1:-1]) 
# # text_total_data = str(text_total_data[0])
# # print(len(text_total_data))
# with open('text_total_data.txt', 'w', encoding = 'utf-8') as f:
# 	f.write(text)
# print(len(text))
# import pickle
# with open('SG_parameters.pickle', 'rb') as f:
#   input_data = pickle.load(f)
# print(input_data)
# data = open('text_total_data.pickle', 'r').read() # should be simple plain text file
# data = ""
# for lists in input_data[0:2]:
#   for word in lists:
#     data += word
# data = data.split(' ')
# chars = list(set(data))
# print(chars)
