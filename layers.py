import numpy as np
from loss import cross_entropy_error
from activation import sigmoid, relu, softmax
from collections import OrderedDict

def numerical_gradient(f,x): #편미분
    h = 1e-4
    grad = np.zeros_like(x)
    if x.ndim == 2: 
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                fx = f(x[i,j]) #자기자신을 미분
                tmp_val = x[i,j]
                x[i,j] = tmp_val + h 
                fxh = f(x[i,j])
                grad[i,j] = (fxh - fx)/h
                x[i,j] = tmp_val
        return grad
    else:
        for i in range(x.size):
            tmp_val = x[i]
            x[i] = tmp_val + h
            fxh1 = f(x[i])
            x[i] = tmp_val - h
            fxh2 = f(x[i])
            grad[i] = (fxh1-fxh2)/2*h
            x[i] = tmp_val
        return grad

    

class TwoLayerNet: 
    def __init__(self,input_size,hidden_size,output_size):
        self.W = {}
        self.W['W1'] = np.random.randn(input_size,hidden_size)
        self.W['b1'] = np.random.randn(hidden_size)
        self.W['W2'] = np.random.randn(hidden_size,output_size)
        self.W['b2'] = np.random.randn(output_size)
        self.loss_val = []
    
    def predict(self,x):
        W1 = self.W['W1'] #가중치: 열별 비중이 다르게 반영해서 백점만점으로 환산하는 경우 각 값에 가중치를 곱하고 그 결과를 합한다
        W2 = self.W['W2']
        b1 = self.W['b1'] #`추정량의 기대값`이 `실제 모수 값`과 차이, 치우침, 약간의 범위차를 만든다
        b2 = self.W['b2']
        
        a1 = np.dot(x,W1) + b1 # 출력값
        z1 = relu(a1) #입력값이 0보다 작으면 0으로 출력, 0보다 크면 입력값 그대로 출력
        a2 = np.dot(z1,W2) + b2
        out = softmax(a2) #나온값이 답과 같을 확률
        return out
    
    def loss(self,x,t):
        y = self.predict(x)
        loss = cross_entropy_error(y,t)
        return loss

    def numerical_gradient(self,x,t):
        f = lambda W: self.loss(x,t) #손실함수: 예측값와 실제값에 대한 오차
        
        grads = {}
        grads['W1'] = numerical_gradient(f, self.W['W1'])
        grads['b1'] = numerical_gradient(f, self.W['b1'])
        grads['W2'] = numerical_gradient(f, self.W['W2'])
        grads['b2'] = numerical_gradient(f, self.W['b2'])
        
        return grads

    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        acc = sum(y == t)/x.shape[0]
        return acc
    
    def train(self,epochs,lr,x,t):
        for epoch in range(epochs):
            grads = self.numerical_gradient(x,t)
            for key in grads.keys():
                self.W[key] -= lr*grads[key]
            self.loss_val.append(self.loss(x,t))

            
            
            
class OneLayer:
    def __init__(self,input_size,output_size):
        self.W = {}
        self.W['W1'] = np.random.randn(input_size,output_size)
        self.W['b'] = np.random.randn(output_size)
    
    def predict(self,x):
        W1, b = self.W['W1'], self.W['b']
        pred = softmax(np.dot(x,W1) + b)
        return pred
    
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
    
    def numerical_gradient(self,x,t):
        y = self.predict(x)
        f = lambda W: cross_entropy_error(y,t)
        grad = {}
        grad['W1'] = numerical_gradient(f,self.W['W1'])
        grad['b'] = numerical_gradient(f, self.W['b'])
        
        return grad
    
    def accuracy(self,x,t):
        y = self.predict(x)
        acc = np.sum(np.argmax(y,axis=1) == np.argmax(t,axis=1))/y.shape[0]
        return acc
    
    def fit(self,x,t,epochs=1000,lr=1e-3,verbos=1):
        for epoch in range(epochs):
            self.W['W1'] = self.W['W1'] - lr*self.numerical_gradient(x,t)['W1']
            self.W['b'] -= lr*self.numerical_gradient(x,t)['b']
            if verbos == 1:
                print("=========== loss ",self.loss(x,t), "======== acc ",self.accuracy(x,t))
                
                

                
                
class MultiLayerNet: #지우기
    def __init__(self,input_size,hidden_size,output_size):
        hidden_size.append(output_size)
        self.W = {}
        self.W = {}
        self.W['Input'] = np.random.randn(input_size,hidden_size[0])
        self.W['Input_b'] = np.random.randn(hidden_size[0])
        for i in range(len(hidden_size)-1):
            w = 'W'+str(i)
            b = 'b'+str(i)
            self.W[w] = np.random.randn(hidden_size[i],hidden_size[i+1])
            self.W[b] = np.random.randn(hidden_size[i+1])      
    
    def predict(self,x):
        j = 0
        for i in range(len(self.W)):
            if j % 2 == 0 and i < (len(self.W)-1):
                x = relu(np.dot(x,self.W[list(self.W.keys())[i]]) + self.W[list(self.W.keys())[i+1]])
            elif j % 2 == 0 and i >= (len(self.W)-1):
                x = (np.dot(x,self.W[list(self.W.keys())[i]]) + self.W[list(self.W.keys())[i+1]])
            j += 1
        return softmax(x)
    
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
    
    def numerical_gradient(self,x,t):
        f = lambda W: self.loss(x,t)
        grads = {}
        for key in self.W.keys():
            grads[key] = numerical_gradient(f,self.W[key])
        return grads
    
    def accuracy(self,x,t):
        y = np.argmax(self.predict(x),axis=1)
        t = np.argmax(t, axis=1)
        acc = np.sum(y==t)/y.size
        return acc
    
    def fit(self,epochs,lr,x,t,verbos=1):
        for epoch in range(epochs):
            for key in self.W.keys():
                self.W[key] -= lr*self.numerical_gradient(x,t)[key]
            if verbos == 1:
                print(epoch,":epoch============== accuracy: ",self.accuracy(x,t),"==========loss :", self.loss(x,t))
                
                
                


                
                
                
#ACTIVATION 함수 4가지                
                
class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
    
class Sigmoid: #함수식 외우기, t값으로 치환하여 계산하는거 혼자 해보기
    def __init_(self):
        self.out = None
    
    def forward(self,x):
        out = 1 / (1+np.exp(-x))
        self.out = out
        return out 
    
    def backward(self,dout):
        dx = dout*self.out*(1-self.out)
        return dx
                  
                
class SoftmaxWithLoss: 
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self,x,t):
        self.y = softmax(x) #self.predict(x)
        self.t = t
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    
    def backward(self,dout=1):
        dx = dout*(self.y-self.t)/self.y.shape[0]
        return dx
    
    
class Affine: #레이어 끝은 cnn이든, 뭐든 무조건 어파인
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self,x): #=predict
        self.x = x
        out = np.dot(x,self.W)+self.b #원래값과 바뀐값의 곱
        
        return out
    
    def backward(self,dout):
        dx = np.dot(dout, self.W.T)    
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        
        return dx
    
    
    
    
    
    
class TwoLayerNet2:     #TwoLayerNet1 수정: 빠르게 나오게 하려고
    def __init__(self,input_size,hidden_size,output_size):
        self.W = {}
        self.W['W1'] = np.random.randn(input_size,hidden_size)
        self.W['b1'] = np.random.randn(hidden_size)
        self.W['W2'] = np.random.randn(hidden_size,output_size)
        self.W['b2'] = np.random.randn(output_size)
        self.loss_val = []
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.W['W1'],self.W['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.W['W2'],self.W['b2'])
        self.loss_val=[]
        self.acc_val=[]
        
        self.lastLayer = SoftmaxWithLoss()
    
    def predict(self,x):
        for layer in self.layers.values(): #x업데이트
            x = layer.forward(x)
        return x
    
    def loss(self,x,t):
        y = self.predict(x)
        # loss = cross_entropy_error(y,t) #미분하려고 구한거(수치적) >> 해석적으로 쓰려고 class만듦 :forwardmax, backwardmax
        loss = self.lastLayer.forward(y,t)
        return loss

    def numerical_gradient(self,x,t):
        f = lambda W: self.loss(x,t)
        
        grads = {}
        grads['W1'] = numerical_gradient(f, self.W['W1'])
        grads['b1'] = numerical_gradient(f, self.W['b1'])
        grads['W2'] = numerical_gradient(f, self.W['W2'])
        grads['b2'] = numerical_gradient(f, self.W['b2'])
        
        return grads
    
    def gradient(self,x,t): #backward시켜주는거 #맨 마지막loss부터 거꾸로 #grad(기울기)를 원래 값에서 빼주면 학습이 되는것이기 때문
        self.loss(x,t)
        dout = 1
        dout = self.lastLayer.backward(dout) #미분값
        layers = list(self.layers.values()) #끝까지 계산했다 다시 back해줘야해서 순서 바꾸기
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout) 
            
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        acc = sum(y == t)/x.shape[0]
        return acc
    
    def fit(self,epochs,lr,x,t):
        for epoch in range(epochs):
            grads = self.gradient(x,t)
            self.W['W1'] -= lr*grads['W1']
            self.W['b1'] -= lr*grads['b1']
            self.W['W2'] -= lr*grads['W2']
            self.W['b2'] -= lr*grads['b2']
            print("epoch",epoch,"==============",self.loss(x,t),"accuracy:=============",self.accuracy(x,t))
            self.loss_val.extend([self.loss(x,t)])
            self.acc_val.extend([np.round(self.accuracy(x,t)),2])
            
            

            
            
            
            
class MultiLayer:
    def __init__(self,input_size,hidden_size,output_size): #레이어 만들기
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.hidden_size.insert(0,self.input_size)
        self.hidden_size.append(self.output_size)
        self.W = {}
        for i in range(len(hidden_size)-1):
            w_key = 'W'+str(i+1)
            b_key = 'b'+str(i+1)
            self.W[w_key] = np.random.randn(hidden_size[i],hidden_size[i+1])
            self.W[b_key] = np.random.randn(hidden_size[i+1])
            
        self.layers = OrderedDict()
        
        for i in range(int(len(self.W)/2-1)):
            j = i*2 
            key1 = 'Affine'+str(i+1)
            key2 = 'Relu'+str(i+1)
            w = list(self.W.keys())[j]
            b = list(self.W.keys())[j+1]
            self.layers[key1] = Affine(self.W[w],self.W[b])
            self.layers[key2] = Relu()
        
        last_num = str(int(len(self.W)/2))
        self.layers['Affine'+last_num] = Affine(self.W['W'+last_num],self.W['b'+last_num])
        self.Lastlayer = SoftmaxWithLoss()
        self.loss_val = []
        self.acc_val = []
    
    #def summary(self):
        
    
    def predict(self,x): #X*w(network(미리구성된 애))
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x,t): #포워드
        y = self.predict(x)
        loss = self.Lastlayer.forward(y,t)
        return loss

    def gradient(self,x,t): #백워드
        self.loss(x,t) #포워드한 값
        dout = 1 #첫번째 값은 무조건 1, 자기자신이니깐
        dout = self.Lastlayer.backward(dout)
        layers = list(self.layers.values()) #손실함수 미분한 값만큼 모으기
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        layer_number = int(len(self.layers.keys())/2)
        
        for i in range(1,layer_number+2): #+2 이유::
            grads['W'+str(i)] = self.layers['Affine'+str(i)].dW
            grads['b'+str(i)] = self.layers['Affine'+str(i)].db
            
        return grads
    
    def accuracy(self,x,t):
        y = np.argmax(self.predict(x),axis=1)
        t = np.argmax(t, axis=1)
        acc = np.sum(y==t)/y.size
        return acc
    

    def fit_sgd(self,epochs,batch_size,lr,x,t,x_val,t_val): #미니배치
        if divmod(x.shape[0],batch_size)[1] > 0:
            batch = divmod(x.shape[0],batch_size)[0] + 1
        else:
            batch = divmod(x.shape[0],batch_size)[0]
        for epoch in range(epochs):
            if epoch == 0:
                start = 0
            end = start + batch_size
            if epoch == epochs-1 and divmod(x.shape[0],batch_size)[1] != 0:
                end = start+divmod(x.shape[0],batch_size)[1]
            x_tmp = x[start:end,:]
            t_tmp = t[start:end,:]
            start = end
            for i in range(batch):
                grads = self.gradient(x_tmp,t_tmp)
            for key in grads.keys():
                self.W[key] -=  lr*grads[key]
            if epoch % 20 == 0:
                print("epoch ",epoch,":val_loss===========",self.loss(x_val,t_val),"val_acc:========",self.accuracy(x_val,t_val))
                self.loss_val.append(self.loss(x_val,t_val))
                self.acc_val.append(np.round(self.accuracy(x_val,t_val),2))
                
                
    def fit_gd(self,epochs,lr,x,t,x_val,t_val): #풀배치
        for epoch in range(epochs):
            grads = self.gradient(x,t)
            for key in grads.keys():
                self.W[key] -=  lr*grads[key]
            if epoch % 20 == 0:
                print("epoch ",epoch,":val_loss===========",self.loss(x_val,t_val),"val_acc:========",self.accuracy(x_val,t_val))
                self.loss_val.append(self.loss(x_val,t_val))
                self.acc_val.append(np.round(self.accuracy(x_val,t_val),2))