import numpy as np
import pickle 
import rnn

from google.colab import files
src = list(files.upload().values())[0]
print(src)
from google.colab import drive
import os 
import sys
from os.path import join
root = '/content/drive/'
drive.mount(root)
path = "My Drive/Colab Notebooks/"   # a custom path. you can change if you want to
main_path = join(root,path)
sys.path.append(main_path)


with open(main_path+'text_total_data.pickle', 'rb') as f:
  input_data = pickle.load(f)
# data = open('text_total_data.pickle', 'r').read() # should be simple plain text file
data = ""
for lists in input_data[0:2]:
  for word in lists:
    data += word
data = data.split(' ')
chars = list(set(data))
print(chars)
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# data = open('input.txt', 'r').read() # should be simple plain text file
# chars = list(set(data))
# data_size, vocab_size = len(data), len(chars)
# char_to_ix = { ch:i for i,ch in enumerate(chars) }
# ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steprob_dict to unroll the RNN for
learning_rate = 1e-1
iteration = 5000
# model parameters
weights = {}
w_x_h = np.random.randn(hidden_size, vocab_size)*0.01 ; weights['w_x_h'] = w_x_h 
w_h_h = np.random.randn(hidden_size, hidden_size)*0.01 ; weights['w_h_h'] = w_h_h
w_h_y = np.random.randn(vocab_size, hidden_size)*0.01 ; weights['w_h_y'] = w_h_y
b_h = np.zeros((hidden_size, 1)) ; weights['b_h'] = b_h
b_y = np.zeros((vocab_size, 1)) ; weights['b_y'] = b_y

def test(h, seed_ix, n):
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(w_x_h, x) + np.dot(w_h_h, h) + b_h)
    y = np.dot(w_h_y, h) + b_y
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mw_x_h, mw_h_h, mw_h_y = np.zeros_like(w_x_h), np.zeros_like(w_h_h), np.zeros_like(w_h_y)
mb_h, mb_y = np.zeros_like(b_h), np.zeros_like(b_y) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
for i in range(iteration):
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  
  if n % 100 == 0:
    sample_ix = test(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix]+' ' for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))


  model = rnn.RnnModule(inputs, targets, hprev, vocab_size, weights)
  model.forward()
  loss, dw_x_h, dw_h_h, dw_h_y, db_h, db_y, hprev, weights = model.backprop()
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: 
    print('iter %d, loss: %f' % (n, smooth_loss)) 
  
  for param, dparam, mem in zip([w_x_h, w_h_h, w_h_y, b_h, b_y], 
                                [dw_x_h, dw_h_h, dw_h_y, db_h, db_y], 
                                [mw_x_h, mw_h_h, mw_h_y, mb_h, mb_y]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) 

  p += seq_length
  n += 1 
with open('rnn_parameters.pickle', 'wb') as f:
  pickle.dump(weights, f)