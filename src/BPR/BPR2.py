import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import csv
import matplotlib.pyplot as plt
import pickle
import math
import time
start = time.time()
mode = 'movielens'
#mode = 'et'
mode1 = 'impli'
#mode1 = 'expli'
p1=[]
p2=[]
p3=[]
p4=[]

average_diff = []

def sigmoid(x):
    return np.exp(-x) / (1 + np.exp(-x))
def predict_a(W,H,user,item):
    if(user==-1 and item==-1):
        ans = np.zeros((len(W),len(H)))
        for i in range(len(W)):
            for j in range(len(H)):
                temp = np.dot(W[int(i),:],H[int(j),:].T)
                if(mode1=='impli'):
                    if temp >= 0:
                        temp = 1.0/(1+np.exp(-temp))
                    else:
                        temp = np.exp(temp)/(1+np.exp(temp))
                ans[i][j] = temp
    elif(item==-1):
        ans = np.zeros((len(H)))
        ans = np.dot(W[int(user),:],H.T)
        ans = np.where(ans>=0,1.0/(1+np.exp(-ans)),np.exp(ans)/(1+np.exp(ans)))
    else:
        ans = np.dot(W[int(user),:],H[int(item),:].T)
        if(mode1=='impli'):
            if ans >= 0:
                ans = 1.0/(1+np.exp(-ans))
            else:
                ans = np.exp(ans)/(1+np.exp(ans))
    return ans
class BPR(object):
    def __init__(self,test,train,trainMatrix,us,it,factor,W,H,l,r,Iter=100):
        self.test = test
        self.train = train
        self.trainMatrix = trainMatrix
        self.us = us
        self.it = it
        self.factor = factor
        self.W = W
        self.H = H
        self.l = l
        self.r = r
        self.Iter = Iter
    def training(self):
        N = len(self.train)
        epochs = self.trainMatrix.nnz
        for epo in range(self.Iter):
            for n in range(epochs):
                u = np.random.randint(self.us)
                itemList = self.trainMatrix.getrowview(u).rows[0]
                if len(itemList) == 0:
                    continue
                i = np.random.choice(itemList)
                self.update(u,i)
            if epo %1 ==0:
                print("epoch : ",epo)
                ans ,ans1= self.cal_in()
                print("a_error(in)",ans,ans1,"a_error(out)",cal_out(self.test,self.W,self.H))
        print("finish")
    def update(self,u,i):
        j = np.random.randint(self.it)
        while self.trainMatrix[u, j] != 0:
            j = np.random.randint(self.it)
        x_pos = self.predict(u,i)
        x_neg = self.predict(u,j)
        xij = -sigmoid(x_pos - x_neg)
        grad_u = self.H[i,] - self.H[j,]
        if(x_pos>0.95 or x_pos<0.05):
            if(np.dot(self.W[u,],self.l * (xij * grad_u + self.W[u,] * self.r)) < 0):
                return
        self.W[u,] -= self.l * (xij * grad_u + self.W[u,] * self.r)
        grad = self.W[u,]
        self.H[i,] -= self.l * (xij * grad + self.H[i,] * self.r)
        self.H[j,] -= self.l * (-xij * grad + self.H[j,] * self.r)
    def predict(self,u,i):
        ans = np.dot(self.W[int(u),:],self.H[int(i),:].T)
        if(mode1=='impli'):
            if ans >= 0:
                ans = 1.0/(1+np.exp(-ans))
            else:
                ans = np.exp(ans)/(1+np.exp(ans))
        return ans
    def cal_in(self):
        error = 0
        mse = 0
        for i in self.train:
            u = int(i[0])
            item = int(i[1])
            y_hat = self.predict(u,item)
            #print(y_hat,i[2])
            error += abs(i[2] - y_hat)
            mse += (i[2]-y_hat)**2
        return error/len(train) , mse/len(train)
    def add_new_rate(self,rating):
        self.trainMatrix[rating[0],rating[1]] = 1
        temp = list(self.train)
        temp.append(rating)
        self.train=np.array(temp)
        itemList = self.trainMatrix.getrowview(int(rating[0])).rows[0]
        for i in range(self.Iter):
            random.shuffle(itemList)
            for j in range(len(itemList)):
                k = itemList[j]
                self.update(int(rating[0]), int(k))
                
        


def cal_out(test,W,H):
    temp = 0
    for i in test:
        utest = int(i[0])
        itest = int(i[1])
        y_hat = predict_a(W,H,utest,itest)
        temp += (i[2] - y_hat)**2
    return temp/len(test)

def predict_top100(W,H,test):
    ans = []
    for i in test:
        temp = np.argsort(-predict_a(W,H,i[0],-1))[:100].tolist()
        temp.insert(0,i[3])
        temp.insert(0,i[0])
        ans.append(temp)
    return ans

if(mode=='movielens'):
    train = pd.read_csv('train.csv',header=None)
    test = pd.read_csv('test.csv',header=None)
    train[:][2] = np.ones((len(train)))
    test[:][2] = np.ones((len(test)))
    train = np.array(train)
    test = np.array(test)
    c = np.r_[train,test]
    us = int(np.max(c[:, 0])+1)
    it = int(np.max(c[:, 1])+1)
    print("Using movielens")

trainMatrix = sp.lil_matrix((us, it))
for i in train:
	trainMatrix[i[0],i[1]] = 1

np.random.seed(7)
f = 10
l = 0.01
r = 0.01
Iter = 100
W = np.random.rand(us, f,)
H = np.random.rand(it, f,)

end = time.time()
print("Start Training : ",end-start)
start = end
Model = BPR(test,train,trainMatrix,us,it,f,W,H,l,r,Iter)
Model.training()
end = time.time()
print("Training cost : ",end-start)
start = end
out1=[]
for i in range(len(test)):
    if i %1000 ==0:
        print("epoch : ",i)
        ans = cal_out(test,Model.W,Model.H)
        print("a_error(out)",ans)
    temp = np.argsort(-predict_a(Model.W,Model.H,test[i][0],-1))[:100].tolist()
    ans = np.where(temp==test[1])
    if(len(ans[0])>0):
        out1.append(ans[0][0])
    else:
        out1.append(0)
    Model.add_new_rate(test[i])
end = time.time()
print("Updating cost : ",end-start)
print("After update",cal_out(test,Model.W,Model.H))


with open('2.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(out1)




