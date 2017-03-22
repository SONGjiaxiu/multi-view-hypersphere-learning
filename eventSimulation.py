__author__ = 'Xian Teng'
# simulate anomalies in region-region networks

import networkx as nx
import numpy as np
import random

# set network attributes
def set_attr(G,z,mu,omega,ntimestamps,F):
	timestamps = np.arange(ntimestamps)
	w = [F * np.exp(-(x-mu)**2/(2*omega))/(np.sqrt(2*omega*np.pi)) for x in timestamps] # temporal traffic flows
	for vex in G.nodes():
		G.node[vex]['topic'] = z # set node attributes
	for edge in G.edges_iter():
		G[list(edge)[0]][list(edge)[1]]['trans'] = w # set edge attributes
	return G

# change node attribute
def change_node_attr(G,node_topic):
	node_list = node_topic.keys()
	for i in np.arange(len(node_list)):
		G.node[node_list[i]]['topic'] = node_topic[node_list[i]]
	return G

# change edge attribute
def change_edge_attr(G,edge_trans,ntimestamps):
	timestamps = np.arange(ntimestamps)
	edge_list = edge_trans.keys()
	for i in np.arange(len(edge_list)):
		F = edge_trans[edge_list[i]]['F']
		mu = edge_trans[edge_list[i]]['mu']
		omega = edge_trans[edge_list[i]]['omega']
		w = [F * np.exp(-(x-mu)**2/(2*omega))/(np.sqrt(2*omega*np.pi)) for x in timestamps]
		G[edge_list[i][0]][edge_list[i][1]]['trans'] = w
	return G

# create a group of normal samples
# output consists of feature matrices
def normal_samples(G,id,N):
	n = G.number_of_nodes()
	e = G.edges()[id]
	ntimestamps = len(G[list(e)[0]][list(e)[1]]['trans'])
	ntopic = len(G.node[id]['topic'])

	feature_matrix = {}
	trans = {} # store normal samples into dictionary
	topic = {}
	target_node = G[id].keys()

	for i in np.arange(N):
		# -------- W matrix ---------- #
		Wout = np.zeros((n,ntimestamps)) # matrix for outflows
		Win = np.zeros((n,ntimestamps)) # matrix for inflows
		for j in target_node:
			Wout[j-1,:] = G[id][j]['trans'] * (1 + np.random.uniform(-1,1,ntimestamps)*0.01) # add 1% noises
			Win[j-1,:] = G[j][id]['trans'] * (1 + np.random.uniform(-1,1,ntimestamps)*0.01)
		W = np.concatenate((Wout,Win),axis=0)
		trans[i] = W
		# -------- Z matrix ---------- #
		Z = np.zeros((ntopic,ntimestamps))
		for j in np.arange(ntimestamps):
			Z[:,j] = G.node[id]['topic'] * (1 + np.random.uniform(-1,1,ntopic)*0.01)
			Z[:,j] = Z[:,j]/sum(Z[:,j]) # scale topic distribution
		topic[i] = Z
	# add trans and topic into a dictionary
	feature_matrix['trans'] = trans 
	feature_matrix['topic'] = topic
	return feature_matrix

# extract trans feature matrix
def extract_trans_feature_matrix(G,id):
	n = G.number_of_nodes()
	e = G.edges()[id]
	ntimestamps = len(G[list(e)[0]][list(e)[1]]['trans'])
	target_node = G[id].keys()
	Wout = np.zeros((n,ntimestamps))
	Win = np.zeros((n,ntimestamps))
	for j in target_node:
		Wout[j-1,:] = G[id][j]['trans']
		Win[j-1,:] = G[j][id]['trans']
	W = np.concatenate((Wout,Win),axis = 0)
	return W

# extract topic feature matrix
def extract_topic_feature_matrix(G,id):
	ntopic = len(G.node[id]['topic'])
	e = G.edges()[id]
	ntimestamps = len(G[list(e)[0]][list(e)[1]]['trans'])
	target_node = G[id].keys()
	Z = np.zeros((ntopic,ntimestamps))
	for j in np.arange(ntimestamps):
		Z[:,j] = G.node[id]['topic']
		Z[:,j] = Z[:,j]/sum(Z[:,j]) # scale topic distribution
	return Z
