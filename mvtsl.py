__author__ = 'Xian Teng'
# multiview time series pattern learning for event detection #

import networkx as nx
import numpy as np



# generate laplacian matrix based on time window s and ntimestamps
def generate_laplacian_matrix(ntimestamps,s):
	weight_matrix = np.zeros((ntimestamps, ntimestamps))
	for i in np.arange(ntimestamps):
		for j in np.arange(ntimestamps):
			if np.abs(i-j) <= np.floor(0.5*s): # if i and j are within a time window
				weight_matrix[i,j] = 1
				weight_matrix[j,i] = 1
	diag_matrix = np.diag(np.sum(weight_matrix,axis=0))
	laplacian_matrix = diag_matrix - weight_matrix
	return laplacian_matrix

# compute objective function
def compute_obj(trans,topic,P_trans,P_topic,Q_trans,Q_topic,Y,lambda2,L):
	nsamples = len(trans)
	obj = 0 # initialize final objective value
	for i in np.arange(nsamples):
		X_trans = trans[i]
		X_topic = topic[i]
		obj += np.linalg.norm(P_trans.transpose().dot(X_trans).dot(Q_trans) - Y) + \
			np.linalg.norm(P_topic.transpose().dot(X_topic).dot(Q_topic) - Y) + \
			np.trace(P_trans.transpose().dot(X_trans).dot(L).dot(X_trans.transpose()).dot(P_trans)) + \
			np.trace(P_topic.transpose().dot(X_topic).dot(L).dot(X_topic.transpose()).dot(P_topic))
	return obj


# multiview discriminantive bilinear projection
def mdbp_optimization(feature_matrix,s,p,q,lambda2,gamma,max_cnt):
	trans = feature_matrix['trans']
	topic = feature_matrix['topic']
	nsamples = len(trans) # no. of samples
	ntrans = len(trans[0]) # no. of feature in trans
	ntopic = len(topic[0]) # no. of feature in topic
	ntimestamps = len(trans[0][0,:]) # no. of timestampes

	# print "nsamples", nsamples
	# print "ntrans", ntrans
	# print "ntopic", ntopic
	# print "ntimestamps", ntimestamps
	
	# normalize samples
	avg_trans = np.zeros((ntrans,ntimestamps))
	avg_topic = np.zeros((ntopic,ntimestamps))
	for i in np.arange(nsamples):
		avg_trans += trans[i]
		avg_topic += topic[i]
	avg_trans = avg_trans/nsamples
	avg_topic = avg_topic/nsamples
	for i in np.arange(nsamples):
		trans[i] -= avg_trans
		topic[i] -= avg_topic
	# generate Laplacian matrix
	L = generate_laplacian_matrix(ntimestamps,s)
	
	# ---- initialize P,Q,Y ---- #
	P_trans = np.random.rand(ntrans,p) # ntrans * p, ntrans > p
	P_topic = np.random.rand(ntopic,p) # ntopic * p
	Q_trans = np.random.rand(ntimestamps,q) # ntimestamps * q, ntimestamps > q
	Q_topic = np.random.rand(ntimestamps,q) # ntimestamps * q
	Y = np.random.rand(p,q) # centroid in latent subspace
	
	# --- orthonormalize P & Q --- #
	P_trans = np.linalg.qr(P_trans)[0] # np.linalg.qr() compute qr decomposition
	P_topic = np.linalg.qr(P_topic)[0]
	Q_trans = np.linalg.qr(Q_trans)[0]
	Q_topic = np.linalg.qr(Q_topic)[0]
	
	# --- compute original objective value --- #
	obj = compute_obj(trans,topic,P_trans,P_topic,Q_trans,Q_topic,Y,lambda2,L)
	# print "obj", obj

	# ---- iterative update ---- #
	for cnt in np.arange(max_cnt):
		# print "cnt", cnt
		# ---- compute partial derivative ---- #
		# trans
		delta_p_trans = np.zeros((ntrans,p))
		delta_q_trans = np.zeros((ntimestamps,q))
		# topic
		delta_p_topic = np.zeros((ntopic,p))
		delta_q_topic = np.zeros((ntimestamps,q))
		# centroid Y in latent space
		delta_y = np.zeros((p,q))

		for i in np.arange(nsamples):
			X_trans = trans[i] # the i-th traning sample
			X_topic = topic[i]

			# --- P & Q in trans view --- #
			delta_p_trans += 2 * X_trans.dot(Q_trans).dot(Q_trans.transpose().dot(X_trans.transpose()).dot(P_trans) - Y.transpose()) \
								+ 2 * lambda2 * X_trans.dot(L).dot(X_trans.transpose()).dot(P_trans)
			
			delta_q_trans += 2 * X_trans.transpose().dot(P_trans).dot(P_trans.transpose().dot(X_trans).dot(Q_trans) - Y)

			# --- P & Q in topic view --- #
			delta_p_topic += 2 * X_topic.dot(Q_topic).dot(Q_topic.transpose().dot(X_topic.transpose()).dot(P_topic) - Y.transpose()) \
								+ 2 * lambda2 * X_topic.dot(L).dot(X_topic.transpose()).dot(P_topic)
			
			delta_q_topic += 2 * X_topic.transpose().dot(P_topic).dot(P_topic.transpose().dot(X_topic).dot(Q_topic) - Y)

			delta_y += - 2 * (P_trans.transpose().dot(X_trans).dot(Q_trans) - Y) - 2 * (P_topic.transpose().dot(X_topic).dot(Q_topic) - Y)

		# update all matrices
		P_trans = P_trans - gamma * delta_p_trans
		P_topic = P_topic - gamma * delta_p_topic
		Q_trans = Q_trans - gamma * delta_q_trans
		Q_topic = Q_topic - gamma * delta_q_topic
		Y = Y - gamma * delta_y

		# orthogonalize P and Q
		P_trans = np.linalg.qr(P_trans)[0]
		P_topic = np.linalg.qr(P_topic)[0]
		Q_trans = np.linalg.qr(Q_trans)[0]
		Q_topic = np.linalg.qr(Q_topic)[0]

		new_obj = compute_obj(trans,topic,P_trans,P_topic,Q_trans,Q_topic,Y,lambda2,L)
		# print "new_obj", new_obj

		if new_obj - obj < -0.001: # if converges
			obj = new_obj
		else:
			result_dict = {'P_trans':P_trans, 'Q_trans':Q_trans, 'P_topic':P_topic, 'Q_topic':Q_topic, 'Y':Y}
			return result_dict

# To be continuted...
# One-class classification algorithm
# def SVDD(X,C):













