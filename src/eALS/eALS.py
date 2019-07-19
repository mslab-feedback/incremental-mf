from __future__ import absolute_import, division, print_function
import os, sys, random, time, math
import numpy as np
import pandas as pd
from scipy.sparse import *
from scipy import *
import tensorflow as tf

import argparse
import numbers # type check in freq_list
if __name__ == '__main__':
	from tqdm import tqdm as tqdm
else:
	from tqdm import tqdm_notebook as tqdm
import multiprocessing
import metrics

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


class my_lil_matrix:
	def __init__(self, shape, fields = ['rating', 'weight']):
		a, b = shape
		self.a = a
		self.b = b
		self.fields = fields
		self.index = [[] for i in range(a)]
		for field in fields:
			setattr(self, field, [[] for i in range(a)])
	def resize(self, shape): # supports only increasing size
		a, b = shape
		if a > self.a:
			self.index.extend([[] for i in range(a - self.a)])
			for field in self.fields:
				getattr(self, field).extend([[] for i in range(a - self.a)])
		self.a = a
		self.b = b
	def add_data(self, rows, cols, data):
		for i in range(len(rows)):
			self.index[rows[i]].append(cols[i])
			for field,d in zip(self.fields, data):
				getattr(self, field)[rows[i]].append(d[i])
class double_lil_matrix:
	def __init__(self, shape, fields = ['rating', 'weight']):
		a, b = shape
		self.row = my_lil_matrix(shape = (a,b), fields = fields)
		self.col = my_lil_matrix(shape = (b,a), fields = fields)
	def resize(self, shape): # supports only increasing size
		a, b = shape
		self.row.resize(shape = (a,b))
		self.col.resize(shape = (b,a))
	def add_data(self, rows, cols, data):
		rows = np.array(rows).astype(int)
		cols = np.array(cols).astype(int)
		self.row.add_data(rows, cols, data)
		self.col.add_data(cols, rows, data)

class freq_list:
	def __init__(self, length, alpha, c0):
		self.data = [0 for _ in range(length)]
		self.sum = 0
		self.alpha = alpha
		self.c0 = c0
	def resize(self, length):
		if len(self.data) < length:
			self.data.extend([0 for _ in range(length - len(self.data))])
	def __len__(self):
		return len(self.data)
	def __setitem__(self, key, value):
		if isinstance(key, numbers.Integral):
			self.sum -= self.data[key]**self.alpha
			self.data[key] = value
			self.sum += self.data[key]**self.alpha
		if isinstance(key, np.ndarray) or isinstance(key, list):
			for i, v in zip(key, value):
				self.sum -= self.data[i]**self.alpha
				self.data[i] = v
				self.sum += self.data[i]**self.alpha
		else:
			raise TypeError("Invalid argument type.")
	def __getitem__(self, key):
		if isinstance(key, numbers.Integral):
			return self.c0 * self.data[key]**self.alpha / max(self.sum, 1)
		elif isinstance(key, slice):
			return self.c0 * np.array([self.data[i] for i in range(len(self.data))[key]])**self.alpha / max(self.sum, 1)
		elif isinstance(key, np.ndarray) or isinstance(key, list):
			return self.c0 * np.array([self.data[i] for i in key])**self.alpha / max(self.sum, 1)
		else:
			raise TypeError("Invalid argument type.")
	def __repr__(self):
		return self.c0 * np.array([self.data[i] for i in range(len(self.data))])**self.alpha / max(self.sum, 1)
	def __iadd__(self, items):
		if isinstance(items, np.ndarray) or isinstance(items, list):
			for i in items:
				self.sum -= self.data[i]**self.alpha
				self.data[i] += 1
				self.sum += self.data[i]**self.alpha
			return self
		else:
			raise TypeError("Invalid argument type.")
		
class eALS:
	def __init__(self, user_num = 1, item_num = 1, K = 10, lmbda = 0.01, alpha = 0.4, w0 = 1, c0 = 512):
		random.seed(1)
		np.random.seed(1)
		self.init_mean = 0
		self.init_std = 0.01    

		self.M = user_num
		self.N = item_num
		self.K = K #vector length

		self.P = np.random.normal(self.init_mean, self.init_std, (self.M, self.K))
		self.Q = np.random.normal(self.init_mean, self.init_std, (self.N, self.K))

		self.lmbda = lmbda #regularization
		self.w0 = w0 #default weight
		self.C = freq_list(self.N, alpha, c0)

		# caches, K * K
		self.Sq = np.zeros((self.K, self.K))
		self.Sp = np.zeros((self.K, self.K))

		# sparse matrix, includes ratings and weights
		self.matrix = double_lil_matrix((self.M, self.N), fields = ['rating', 'weight'])

	def update_user_(self, u):
		Ru = np.array(self.matrix.row.index[u])
		if len(Ru) == 0:
			return self.P[u,:]
		Rutmp = np.array(self.matrix.row.rating[u])
		Ctmp = self.C[Ru]
		Wtmp = np.array(self.matrix.row.weight[u])
		Qtmp = self.Q[Ru,:]
		Rhat = np.matmul((self.P[u,:]).reshape(1, self.K), Qtmp.reshape((len(Ru), self.K)).T)[0, :]
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
	def update_item_(self, i):
		Ri = np.array(self.matrix.col.index[i])
		if len(Ri) == 0:
			return self.Q[i,:]
		Ritmp = np.array(self.matrix.col.rating[i])
		Wtmp = np.array(self.matrix.col.weight[i])
		Ptmp = self.P[Ri,:]
		Rhat = np.matmul(Ptmp, (self.Q[i,:]).reshape(1, self.K).T)[:,0]
		for f in range(self.K):
			Ptmpf = Ptmp[:,f]
			Rhatf = Rhat - (Ptmpf * self.Q[i,f])
			###
			nominator = np.sum((Wtmp * Ritmp - \
					(Wtmp - self.C[i]) * Rhatf) * Ptmpf) - \
					self.C[i] * (np.sum(self.Q[i,:] * self.Sp[:,f]) - \
					self.Q[i,f] * self.Sp[f,f])

			denominator = self.lmbda + self.C[i] * self.Sp[f,f] + \
					np.sum((Wtmp - self.C[i]) * Ptmpf**2)
			self.Q[i,f] = nominator / denominator
			###
			Rhat = Rhatf + (self.P[Ri,f] * self.Q[i,f])
		return self.Q[i,:]
	def update_user(self, u):
		self.Sp -= np.matmul(self.P[u, :].reshape(self.K, 1), self.P[u,:].reshape(1, self.K))
		self.P[u,:] = self.update_user_(u)
		self.Sp += np.matmul(self.P[u, :].reshape(self.K, 1), self.P[u,:].reshape(1, self.K))
	def update_item(self, i):
		self.Sq -= self.C[i] * np.matmul(self.Q[i, :].reshape(self.K, 1), self.Q[i,:].reshape(1, self.K))
		self.Q[i,:] = self.update_item_(i)
		self.Sq += self.C[i] * np.matmul(self.Q[i, :].reshape(self.K, 1), self.Q[i,:].reshape(1, self.K))
	def recompute_Sp(self):
		self.Sp = np.matmul(self.P.T, self.P).reshape((self.K, self.K))
	def recompute_Sq(self):
		self.Sq = np.matmul(self.Q.T * self.C[:].reshape((1, len(self.C))), self.Q).reshape((self.K, self.K))

	def resize(self, N, M):
		new_M = max(self.M, M)
		old_M = self.M
		new_N = max(self.N, N)
		old_N = self.N

		if new_M > old_M:
			new_num = new_M - old_M
			self.P = np.concatenate((self.P, 
				np.random.normal(self.init_mean, self.init_std, (new_num, self.K))), axis = 0)
			self.recompute_Sp()
		if new_N > old_N:
			new_num = new_N - old_N
			self.Q = np.concatenate((self.Q, 
				np.random.normal(self.init_mean, self.init_std, (new_num, self.K))), axis = 0)
			self.C.resize(new_N)

		self.matrix.resize((new_M, new_N))

		self.M = new_M
		self.N = new_N

	def update(self, user, item, rating = 1, weight = 1):
		self.add_data([user], [item], [rating], [weight])

		self.update_user(user)
		self.update_item(item)
	def add_data(self, users, items, ratings = None, weights = None):
		if ratings is None:
			ratings = [1 for _ in users]
		if weights is None:
			weights = [self.w0 for _ in users]
		new_M = max(users) + 1
		new_N = max(items) + 1

		self.resize(N = new_N, M = new_M)
		self.matrix.add_data(users, items, (ratings, weights))
		# update c and therefore Sq
		uniq = np.unique(items)
		self.Sq -= np.matmul(self.Q[uniq, :].reshape((len(uniq), self.K)).T * self.C[uniq].reshape((1, len(uniq))), self.Q[uniq, :].reshape((len(uniq), self.K))).reshape((self.K, self.K))
		old_sum = self.C.sum
		self.C += items 
		self.Sq *= old_sum/self.C.sum
		self.Sq += np.matmul(self.Q[uniq, :].reshape((len(uniq), self.K)).T * self.C[uniq].reshape((1, len(uniq))), self.Q[uniq, :].reshape((len(uniq), self.K))).reshape((self.K, self.K))

	def fit(self, update_mode = False, nprocs=multiprocessing.cpu_count(), prog_bar = False):
		# update user vectors in parallel
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

	def predict_top_k_(self, user, k = 100, gpu = False):
		new_M = max(self.M, user + 1)
		if new_M > self.M:
			self.resize(self.N, new_M)
		if gpu:
			product = tf.matmul(self.P[user, :].reshape((1, self.K)), self.Q.T).numpy()
			tf.set_random_seed(1)
		else:
			product = np.matmul(self.P[user, :], self.Q.T)
		order = np.argsort(-product)
		return order[:k]
	def predict_top_k(self, users, k = 100, gpu = False):
		return np.array([self.predict_top_k_(u, k, gpu) for u in tqdm(users, desc = 'predicting')])
	def evaluate(self, users, items, k = 100, gpu = False):
		predictions = self.predict_top_k(users)
		hr = metrics.get_HR(predictions, items).mean()
		return hr

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
if __name__ == '__main__':
	tf.enable_eager_execution()
	parser = argparse.ArgumentParser()
	parser.add_argument('-T', '--train', type=str)
	parser.add_argument('-t', '--test', type=str)
	parser.add_argument('-o', '--online', action='store_true')
	parser.add_argument('-e', '--epoch', type=int, default=100)
	parser.add_argument('-c', '--cpu', type=int, default=10)
	parser.add_argument('-d', '--dir', type=str, default='.')

	parser.add_argument('-n', '--no_train', action='store_true')
	parser.add_argument('-p', '--predict', action='store_true')
	'''
	parser.add_argument('-T', '--train', type=str, dest='train')
	parser.add_argument('-t', '--test', type=str, dest='test')
	parser.add_argument('-o', '--online', action='store_true', dest='online')
	parser.add_argument('-e', '--epoch', type=int, default=100, dest='epoch')
	'''
	args = parser.parse_args()	
	print ("training file: {}, testing file: {}, online: {}, epoch: {}, cpu: {}, dir: {}".format( \
			args.train, args.test, args.online, args.epoch, args.cpu, args.dir))
	print ("no train: {}, do predict: {}".format( \
			args.no_train, args.predict))
	Pfname = os.path.join(args.dir, "P.txt")
	Qfname = os.path.join(args.dir, "Q.txt")
	predfname = os.path.join(args.dir, "prediction.txt")
	print ("model file names: {}, {}, prediction file names: {}".format(Pfname, Qfname, predfname))
	train_df = pd.read_csv(args.train, names=['userid', 'itemid', 'rating', 'weight'])
	test_df = pd.read_csv(args.test, names=['userid', 'itemid', 'rating', 'weight'])
	model = eALS(K = 10)
	model.add_data(train_df['userid'].values.astype(int), train_df['itemid'].values.astype(int), train_df['rating'].values, train_df['weight'].values)
	model.recompute_Sp()
	model.recompute_Sq()
	if not args.no_train:
		for epoch in tqdm(range(args.epoch), desc = 'training'):
			model.fit(nprocs = args.cpu)
		np.savetxt(Pfname, model.P)
		np.savetxt(Qfname, model.Q)
	if args.predict or args.online:
		model.P = np.loadtxt(Pfname)
		model.Q = np.loadtxt(Qfname)
		model.recompute_Sp()
		model.recompute_Sq()
		if args.online:
			predictions = []
			model.resize(max(model.M, np.max(test_df['userid'].values) + 1),
					max(model.N, np.max(test_df['itemid'].values) + 1))
			for i in tqdm(range(len(test_df['userid'])), desc = 'online'):
				prediction = model.predict_top_k_(test_df['userid'][i], gpu = False)
				where = np.where(prediction == test_df['itemid'][i])[0]
				res = 0 if len(where) == 0 else where[0] + 1
				predictions.append(res)
				model.update(test_df['userid'][i], 
						test_df['itemid'][i], 
						test_df['rating'][i], 
						test_df['weight'][i])
		else:
			predictions = model.predict_top_k(test_df['userid'].values, gpu = False)
		pd.DataFrame(predictions).to_csv(predfname, index = False, header = False)
