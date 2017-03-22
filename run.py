__author__ = 'Xian Teng'
# multi-view time series learning algorithm

import networkx as nx
import numpy as np
import eventSimulation
import mvtsl

G=nx.DiGraph()
G.add_edges_from([(1,2), (1,5),
	(2,1), (2,3), (2,6),
	(3,2), (3,4), (3,7),
	(4,3), (4,8),
	(5,1), (5,6), (5,9),
	(6,2), (6,5), (6,7), (6,10),
	(7,3), (7,6), (7,8), (7,11),
	(8,4), (8,7), (8,12),
	(9,5), (9,10),
	(10,6), (10,9), (10,11),
	(11,7), (11,10), (11,12),
	(12,8), (12,11)])

F = 500
z = [0.05,0.10,0.35,0.4,0.1] # topic distribution
ntimestamps = 12 # time stamps
mu = 6 # parameter for traffic flows
omega = 5

G = eventSimulation.set_attr(G,z,mu,omega,ntimestamps,F) # set node and edge attributes

id = 7 # target region
N = 50 # no. of samples

# generate N samples
# output is dictionary storing 'trans' and 'topic'
feature_matrix = eventSimulation.normal_samples(G,id,N)
trans = feature_matrix['trans']
topic = feature_matrix['topic']
# print trans[0]
# print topic[0]

# define parameters
gamma = 0.001
s = 2 # time window
p = 3 # feature dimension
q = 5 # time dimension
max_cnt = 1000
lambda2 = 0.5
result_dict = mvtsl.mdbp_optimization(feature_matrix,s,p,q,lambda2,gamma,max_cnt)
P_trans = result_dict['P_trans']
# print "P_trans", P_trans
Q_trans = result_dict['Q_trans']
# print "Q_trans", Q_trans
P_topic = result_dict['P_topic']
# print "P_topic", P_topic
Q_topic = result_dict['Q_topic']
# print "Q_topic", Q_topic
Y = result_dict['Y']
# print "Y", Y

print "----- transporation samples -----"
for i in np.arange(N):
	print i, np.linalg.norm(P_trans.transpose().dot(trans[i]).dot(Q_trans) - Y)
print "----- transporation samples -----"
for i in np.arange(N):
	print i, np.linalg.norm(P_topic.transpose().dot(topic[i]).dot(Q_topic) - Y)

# ---------- insert anomaly ----------- #
node_topic = {1:(0.35,0.40,0.05,0.10,0.10), 2:(0.35,0.40,0.05,0.10,0.10), 
				6:(0.35,0.40,0.05,0.10,0.10), 7:(0.35,0.40,0.05,0.10,0.10), 
				11:(0.35,0.40,0.05,0.10,0.10), 12:(0.35,0.40,0.05,0.10,0.10)}
edge_trans = {(1,2):{'F':300, 'mu':4, 'omega':7},
				(2,1):{'F':600, 'mu':2.5, 'omega':4},
				(2,6):{'F':300, 'mu':8, 'omega':6},
				(6,2):{'F':800, 'mu':2, 'omega':1.0},
				(6,7):{'F':100, 'mu':8, 'omega':6},
				(7,6):{'F':1000, 'mu':1, 'omega':0.2},
				(7,11):{'F':800, 'mu':1, 'omega':0.5},
				(11,7):{'F':100, 'mu':9, 'omega':7},
				(11,12):{'F':600, 'mu':2, 'omega':1.0},
				(12,11):{'F':300, 'mu':8.5, 'omega':5}}

G = eventSimulation.change_node_attr(G,node_topic)
G = eventSimulation.change_edge_attr(G,edge_trans,ntimestamps)
W = eventSimulation.extract_trans_feature_matrix(G,id)
Z = eventSimulation.extract_topic_feature_matrix(G,id)

print "----- anomaly -----"
print np.linalg.norm(P_trans.transpose().dot(W).dot(Q_trans) - Y)
print np.linalg.norm(P_topic.transpose().dot(Z).dot(Q_topic) - Y)
