from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import *
from scipy import *
import random, time

from tqdm import tqdm_notebook as tqdm

import multiprocessing

import sys

has_tf = 'tensorflow' in sys.modules
if has_tf:
    import tensorflow as tf
    tf.enable_eager_execution()

# https://stackoverflow.com/a/16071616
def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    if nprocs == 1:
        return [f(x) for x in X]
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res, key = lambda x:x[0])]

init_mean = 0
init_std = 0.01    
class eALS:
    def __init__(self, user_num = 1, item_num = 1, K = 10, lmbda = 0.01, alpha = 0.4, w0 = 1, c0 = 512):

        random.seed(1)
        np.random.seed(1)
        self.M = user_num
        self.N = item_num
        self.K = K

        self.P = np.random.normal(init_mean, init_std, (self.M, self.K))
        self.Q = np.random.normal(init_mean, init_std, (self.N, self.K))

        self.lmbda = lmbda
        self.alpha = alpha
        self.W = dok_matrix((self.M, self.N), dtype=float32)
        self.w0 = w0
        self.c0 = c0
        self.C = np.zeros((self.N,))
        # caches, K * K
        self.Sq = np.zeros((self.K, self.K))
        self.Sp = np.zeros((self.K, self.K))
        # R, use sparse matrix
        self.R = dok_matrix((self.M, self.N), dtype=float32)
        self.Ru = self.R.tocsr()
        self.Ri = self.R.tocsc()
        
    def update_user_(self, u):
        Ru = self.Ru[u,:].nonzero()[1]
        Rutmp = self.R[u,Ru].A
        Ctmp = self.C[Ru]
        Wtmp = self.W[u,Ru].A 
        Qtmp = self.Q[Ru,:]
        Rhat = np.matmul((self.P[u,:]).reshape(1, self.K), Qtmp.T)[:]

        for f in range(self.K):
            Qtmpf = Qtmp[:,f]
            Rhatf = Rhat - self.P[u,f] * Qtmpf
            ###
            nominator = np.sum((Wtmp * Rutmp - \
                (Wtmp - Ctmp) * Rhatf) * Qtmpf) - \
                (np.sum(self.P[u,:] * self.Sq[:,f]) - \
                self.P[u,f] * self.Sq[f,f])
            denominator = self.lmbda + self.Sq[f,f] + \
                np.sum((Wtmp - Ctmp) * Qtmpf**2)
            self.P[u,f] = nominator / denominator
            ###
            Rhat = Rhatf + self.P[u,f] * self.Q[Ru,f]
        return self.P[u,:]
    def update_user(self, u, update_Sp = False):
        if update_Sp:
            self.Sp -= np.matrix(self.P[u]) * np.matrix(self.P[u]).T
        
        self.P[u,:] = self.update_user_(u)
        
        if update_Sp:
            self.Sp += np.matrix(self.P[u]) * np.matrix(self.P[u]).T
    def update_item_(self, i):
        Ri = self.Ri[:,i].nonzero()[0]
        Wtmp = self.W[Ri,i].A
        Ritmp = self.R[Ri,i].A
        Ptmp = self.P[Ri,:]
        Rhat = np.matmul(Ptmp, (self.Q[i,:]).reshape(1, self.K).T)[:]
        #update each factor
        for f in range(self.K):
            Ptmpf = Ptmp[:,f]
            Rhatf = Rhat - (Ptmpf * self.Q[i,f]).reshape(len(Ri), 1)
            ###
            nominator = np.sum((Wtmp * Ritmp - \
                (Wtmp - self.C[i]) * Rhatf) * Ptmpf) - \
                self.C[i] * (np.sum(self.Q[i,:] * self.Sp[:,f]) - \
                            self.Q[i,f] * self.Sp[f,f])
                        
            denominator = self.lmbda + self.C[i] * self.Sp[f,f] + \
                np.sum((Wtmp - self.C[i]) * Ptmpf**2)
            self.Q[i,f] = nominator / denominator
            ###
            Rhat = Rhatf + (self.P[Ri,f] * self.Q[i,f]).reshape(len(Ri), 1)
        return self.Q[i,:]
    def update_item(self, i, update_Sq = False):
        if update_Sq:
            self.Sq -= self.C[i] * np.matrix(self.Q[i,:]) * np.matrix(self.Q[i,:]).T
        
        self.Q[i,:] = self.update_item_(i)
        
        if update_Sq:
            self.Sq += self.C[i] * np.matrix(self.Q[i,:]) * np.matrix(self.Q[i,:]).T
    def recompute_Sp(self):
        self.Sp = np.matmul(self.P.T, self.P).reshape((self.K, self.K))
    def recompute_Sq(self):
        self.Sq = np.matmul(self.Q.T * self.C.reshape((1, len(self.C))), self.Q).reshape((self.K, self.K))
        
    def resize(self, N, M):
        new_M = max(self.M, M)
        old_M = self.M
        new_N = max(self.N, N)
        old_N = self.N
        
        if new_M > old_M:
            new_num = new_M - old_M
            self.P = np.concatenate((self.P, 
                np.random.normal(init_mean, init_std, (new_num, self.K))), axis = 0)
            #self.Ru = self.Ru + [set([]) for _ in range(new_num)]
            self.recompute_Sp()
        if new_N > old_N:
            new_num = new_N - old_N
            self.Q = np.concatenate((self.Q, 
                np.random.normal(init_mean, init_std, (new_num, self.K))), axis = 0)
            #self.Ri = self.Ri + [set([]) for _ in range(new_num)]
            self.C = np.array(self.C.tolist() + [0 for _ in range(old_N, new_N)])
            self.recompute_Sq()
        
        self.W.resize((new_M, new_N))
        self.R.resize((new_M, new_N))
        self.Ru.resize((new_M, new_N))
        self.Ri.resize((new_M, new_N))
        
        self.M = new_M
        self.N = new_N
    def update_c(self):
        freq = np.diff(self.Ri.indptr).reshape((self.N,))
        freq_alpha = freq**self.alpha 
        freq_alpha_sum = np.sum(freq_alpha)
        self.C = self.c0 * freq_alpha/freq_alpha_sum
        self.recompute_Sq()
    def update(self, user, item, rating, new_weight = None, update_c = False):
        self.resize(N = item + 1, M = user + 1)
        self.W[user,item] = new_weight if new_weight else self.w0
        self.R[user,item] = rating
        self.Ru[user,item] = rating
        self.Ri[user,item] = rating

        if update_c:
            self.update_c()
        
        self.update_user(user, True)
        self.update_item(item, True)
    def fit(self, interactons, update_mode = False, nprocs=multiprocessing.cpu_count(), prog_bar = False):
        
        users, items, ratings = interactons
        
        new_M = max(users) + 1
        new_N = max(items) + 1
        
        self.resize(N = new_N, M = new_M)
        self.W[users,items] = self.w0
        self.R[users,items] = ratings

        self.Ru = self.R.tocsr()
        self.Ri = self.R.tocsc()
        
        self.update_c()
        # training
        
        # update user vectors in parallel
        #for u in tqdm(range(self.M)):
        #    self.update_user(u) 

        if prog_bar:
            self.P = np.array(parmap(self.update_user_, tqdm(range(self.M), desc = 'fit users'), nprocs=nprocs))
        else:
            self.P = np.array(parmap(self.update_user_, range(self.M), nprocs=nprocs))
        self.recompute_Sp()
        
        # update item vectors in parallel
        if prog_bar:
            self.Q = np.array(parmap(self.update_item_, tqdm(range(self.N), desc = 'fit items'), nprocs=nprocs))
        else:
            self.Q = np.array(parmap(self.update_item_, range(self.N), nprocs=nprocs))
        self.recompute_Sq()
        
    def predict_top_k_(self, user, k = 100, gpu = has_tf):
        if gpu:
            product = np.matmul(self.P[user, :], self.Q.T)
        else:
            product = tf.matmul(self.P[user, :], self.Q.T).numpy()
        order = np.argsort(-product)
        '''
        if in_train == False:
            order = [idx for idx in order if idx \
                     not in self.Ru[user,:].nonzero()[1]]
        
        '''
        return order[:k]
    def predict_top_k(self, users, k = 100, gpu = has_tf):
        new_M = max(self.M, max(users) + 1)
        old_M = self.M
        self.resize(self.N, new_M)
        predictions = [self.predict_top_k_(u, k, gpu) for u in tqdm(users, desc = 'predicting')]
        if gpu:
            tf.set_random_seed(1) #reset kernel cache
        return np.array(predictions)
        
    def predict_ratings(self, pairs):
        users, items = pairs
        new_M = max(self.M, max(users) + 1)
        new_N = max(self.N, max(items) + 1)
        self.resize(new_N, new_M)
        ratings = [np.dot(model.P[u,:], model.Q[i,:]) for u, i in zip(users, items)]
        return ratings


def get_interactions_from_df(df, user_label, item_label,
                             rating_label = None):
    users = df[user_label].values
    items = df[item_label].values
    ratings = np.ones(len(df))
    if rating_label != None:
        ratings = df[rating_label].values
    return users, items, ratings
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))
