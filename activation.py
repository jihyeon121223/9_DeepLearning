import numpy as np

def sigmoid(x): #분류값2개일때, y가 한줄일때
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(x): #분류값3개 이상일 때
    c = np.max(x,axis=1).reshape(-1,1)
    a = x - c
    return np.exp(a)/np.sum(np.exp(a),axis=1).reshape(-1,1)

