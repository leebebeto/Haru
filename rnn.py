import numpy as np

class RnnModule():
  def __init__(self, inputs, targets, hprev, vocab_size, weights):
    self.inputs = inputs 
    self.vocab_size = vocab_size 
    self.x_dict = {}
    self.h_dict = {}
    self.y_dict = {}
    self.prob_dict = {}
    self.weights = weights
    self.w_x_h = weights['w_x_h']
    self.w_h_h = weights['w_h_h']
    self.w_h_y = weights['w_h_y']
    self.b_h = weights['b_h']
    self.b_y = weights['b_y']
    self.h_dict[-1] = hprev
    self.targets = targets
    self.loss = 0 

  def forward(self):
    for t in range(len(self.inputs)):
      self.x_dict[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
      self.x_dict[t][self.inputs[t]] = 1
      self.h_dict[t] = np.tanh(np.dot(self.w_x_h, self.x_dict[t]) + np.dot(self.w_h_h, self.h_dict[t-1]) + self.b_h) # hidden state
      self.y_dict[t] = np.dot(self.w_h_y, self.h_dict[t]) + self.b_y # unnormalized log probabilities for next chars
      self.prob_dict[t] = np.exp(self.y_dict[t]) / np.sum(np.exp(self.y_dict[t])) # probabilities for next chars
      self.loss += -np.log(self.prob_dict[t][self.targets[t],0]) # softmax (cross-entropy loss)

  def backprop(self):
    dw_x_h, dw_h_h, dw_h_y = np.zeros_like(self.w_x_h), np.zeros_like(self.w_h_h), np.zeros_like(self.w_h_y)
    db_h, db_y = np.zeros_like(self.b_h), np.zeros_like(self.b_y)
    dhnext = np.zeros_like(self.h_dict[0])
    for t in reversed(range(len(self.inputs))):
      dy = np.copy(self.prob_dict[t])
      dy[self.targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
      dw_h_y += np.dot(dy, self.h_dict[t].T)
      db_y += dy
      dh = np.dot(self.w_h_y.T, dy) + dhnext # backprop into h
      dhraw = (1 - self.h_dict[t] * self.h_dict[t]) * dh # backprop through tanh nonlinearity
      db_h += dhraw
      dw_x_h += np.dot(dhraw, self.x_dict[t].T)
      dw_h_h += np.dot(dhraw, self.h_dict[t-1].T)
      dhnext = np.dot(self.w_h_h.T, dhraw)
    for dparam in [dw_x_h, dw_h_h, dw_h_y, db_h, db_y]:
      np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return self.loss, dw_x_h, dw_h_h, dw_h_y, db_h, db_y, self.h_dict[len(self.inputs)-1], self.weights 

