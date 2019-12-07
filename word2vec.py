from google.colab import drive
import torch
import random
from random import shuffle
from collections import Counter
import argparse
import numpy as np
import re
import pickle
# hyper parameters
# skip gram - learning rate : 0.005 / iteration : 50000
# cbow - learning rate : 0.1 / iteration : 50000
drive.mount('/gdrive')
def getRandomContext(corpus, C=5):
    wordID = random.randint(0, len(corpus) - 1)
    
    context = corpus[max(0, wordID - C):wordID]
    if wordID+1 < len(corpus):
        context += corpus[wordID+1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]

    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)



# V = 3971 / D = 64
def Skipgram(centerWord, contextWord, inputMatrix, outputMatrix):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWord : Index of a contextword (type:int)                       #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################

    # setting one-hot encoding
    centerWordVector = []
    contextWordVector = []
    for i in range(inputMatrix.shape[0]):
        if i == centerWord:
            centerWordVector.append(1)
        if i == contextWord:
            contextWordVector.append(1)
        else:
            centerWordVector.append(0)
            contextWordVector.append(0)
    centerWordVector = torch.FloatTensor(centerWordVector)
    contextWordVector = torch.FloatTensor(contextWordVector)
    #feed forward 
    hidden_layer = torch.matmul(torch.t(inputMatrix),centerWordVector)
    output_layer = torch.matmul(torch.t(outputMatrix),hidden_layer)    
    e = torch.exp(output_layer)
    final_layer = e / torch.sum(e)

    loss = None
    grad_emb = None
    grad_out = None

    loss = -torch.log(final_layer[contextWord]+1e-7)
    dfinal_layer = final_layer
    dfinal_layer[contextWord] -= 1
    grad_out = torch.from_numpy(np.outer(hidden_layer.numpy(), dfinal_layer.numpy()))    
    grad_emb = torch.from_numpy(np.outer(centerWordVector.numpy(), np.dot(outputMatrix,dfinal_layer.numpy().T)))
    return loss, grad_emb, grad_out


def CBOW(centerWord, contextWords, inputMatrix, outputMatrix):
    # setting one-hot encoding
    centerWordVector = []
    contextWordVector = []
    for i in range(inputMatrix.shape[0]):
        if i == centerWord:
            centerWordVector.append(1)
        else:
            centerWordVector.append(0)

    for i in range(inputMatrix.shape[0]):
        if i in contextWords:
            contextWordVector.append(1)
        else:
            contextWordVector.append(0)
    
    
    centerWordVector = torch.FloatTensor(centerWordVector)
    contextWordVector = torch.FloatTensor(contextWordVector)
    contextWordVector /= len(contextWords)

    hidden_layer = torch.matmul(torch.t(inputMatrix),contextWordVector)
    output_layer = torch.matmul(torch.t(outputMatrix),hidden_layer)
    e = torch.exp(output_layer)
    final_layer = e / torch.sum(e)

    loss = None
    grad_emb = None
    grad_out = None

    loss = -(torch.log(final_layer[centerWord])+1e-7)
    dfinal_layer = final_layer
    dfinal_layer[centerWord] -= 1
    grad_out = torch.from_numpy(np.outer(hidden_layer.numpy(), dfinal_layer.numpy()))    
    grad_emb = torch.from_numpy(np.outer(centerWordVector.numpy(), np.dot(outputMatrix,dfinal_layer.numpy().T)))

    return loss, grad_emb, grad_out

def word2vec_trainer(corpus, word2ind, mode="CBOW", dimension=100, learning_rate=0.1, iteration=50):

# Xavier initialization of weight matrices
    W_emb = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    W_out = torch.randn(dimension, len(word2ind)) / (dimension**0.5)
    window_size = 5
    losses=[]
    for i in range(iteration):
        #Training word2vec using SGD
        centerword, context = getRandomContext(corpus, window_size)
        centerInd = word2ind[centerword]
        contextInds = [word2ind[word] for word in context]
        if mode=="CBOW":
            L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out)
            W_emb -= learning_rate*G_emb
            W_out -= learning_rate*G_out
            losses.append(L.item())

        elif mode=="SG":
            for contextInd in contextInds:
                L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out)
                W_emb -= learning_rate*G_emb
                W_out -= learning_rate*G_out
                losses.append(L.item())
                
        else:
            print("Unknown mode : "+mode)
            exit()

        if i%50==0:
            print(i)
            avg_loss=sum(losses)/len(losses)
            print("Loss : %f" %(avg_loss,))
            losses=[]

    return W_emb, W_out


# simple function for printing similar words 
def sim(testword, word2ind, ind2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    wi = word2ind[testword]
    inputVector = matrix[wi].reshape(1,-1)/length[wi]
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)
    
    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()


def main():
    # parser = argparse.ArgumentParser(description='Word2vec')
    # parser.add_argument('mode', metavar='mode', type=str,
    #                     help='"SG" for skipgram, "CBOW" for CBOW')
    # parser.add_argument('part', metavar='partition', type=str,
    #                     help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    # args = parser.parse_args()
    # mode = args.mode
    # part = args.part


    # colab이 arg parsing을 지원하지 않아서 CBOW나 SG을 직접 지정을 했습니다! (arg parsing과 같은 효과라고 판단되어서 코드를 수정을 했습니다)
    part = "part"
    mode = "CBOW"
    # mode = "SG"


    #Load and tokenize corpus
    print("loading...")
    text_data = open('/gdrive/My Drive/Colab Notebooks/text_total_data.txt',mode='r').readlines()
    text = ""
    for i in range(len(text_data)):
        text += text_data[i][:len(text_data[i])]
    print('123',len(text))

    input_data = ""
    for lists in text:
        for word in lists:
            input_data += word 
    corpus = input_data.split()
    frequency = Counter(corpus)
    processed = []
    #Discard rare words
    for word in corpus:
        if frequency[word]>6:
            processed.append(word)
    vocabulary = set(processed)
    for words in vocabulary:
        words.lower()
    #Assign an index number to a worda
    word2ind = {}
    word2ind[" "]=0
    i = 1
    for word in vocabulary:
        word2ind[word] = i
        i+=1

    ind2word = {}
    for k,v in word2ind.items():
        ind2word[v]=k

    print("Vocabulary size")
    print(len(word2ind))
    print()

    learning_rate = 0.025
    iteration = 50000

    #Training section
    # W_emb, W_out = word2vec_trainer(processed, word2ind, mode=mode, dimension=64, learning_rate=0.05, iteration=60000)
    W_emb, W_out = word2vec_trainer(processed, word2ind, mode=mode, dimension=64, learning_rate=0.025, iteration=50000)
    W = [W_emb, W_out, word2ind, ind2word]
    #Print similar words
    testwords = ["Obama", "senator", "white", "house", "battle", "University", "Australia", "NASA", "society", "spot"]
    for tw in testwords:
        sim(tw,word2ind,ind2word,W_emb)
    with open('/gdrive/My Drive/Colab Notebooks/'+mode+'_'+str(learning_rate)+'_'+str(iteration)+'_parameters_w.pickle', 'wb') as f:
        pickle.dump(W,f)



main()