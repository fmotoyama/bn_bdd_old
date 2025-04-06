# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 14:36:21 2021

@author: f.motoyama
"""
from graphviz import Digraph
import os
import BooleanNetwork


#f_dict = {2: [[2, 35]], 14: [[14]], 18: [[-2, -14, -18, -35], [-2, -14, 18], [-2, 14, 18], [2, 18, 35]], 20: [[20, 35]], 24: [[-2, -14, -24, 35], [-2, 14, -18, -24], [-2, 14, -18, 24, -35], [-2, 14, 18, -24, 35], [2, -24], [2, 24, -35]], 35: [[-24], [24, -35]]}
f_dict = {1:0, 2:[[1,4]], 3:[[2]], 4:[[3]], 5:[[2]]}
f_dict = {1:[[1],[4]],2:[[1],[2]],3:[[-2,3,4]],4:[[-1],[4]]}
f_dict = {15: [[-15, -38], [15]], 19: [[-15], [15, -19, -33], [15, -19, 33, 38], [15, 19, 38]], 33: [[-15, -33, 38], [-15, 33], [15]], 38: [[-15, 38], [15]]}
f_dict = {1: [[3]], 2: [[1,3]], 3: [[1,2]]}
parent = BooleanNetwork.BooleanNetwork.f_dict_to_parent(f_dict)

red = [[38,15],[15,33]]
blue = [[38,33]]


_type = 'sfdp'
G = Digraph(format='png', engine=_type)
#fontsize = 14, width=0.75
G.attr('node', shape='circle')
G.attr('graph', overlap = '0:')#, splines = 'curved'
G.attr('edge', arrowsize = '0.75')


for key in parent:
    for p in parent[key]:
        if [p,key] in red:
            c = 'red'
            l = str(red.index([p,key]) + 1)
            l = ''
        elif [p,key] in blue:
            c = 'blue'
            l = ''
        else:
            c = 'black'
            l = ''
        G.edge(str(p),str(key), color=c, label = l)



#図を保存
G.render('./figure/BN_')
os.remove('./figure/BN_')


"""
import BooleanNetwork
import drawset
import copy

B = BooleanNetwork.BN()
#BN = B.get_BN(f_dict)
Var = B.get_Var(f_dict.keys())

BN = {}
BN = copy.copy(Var)
BN[15] = B.calc_f(f_dict[15],BN)
BN[33] = B.calc_f(f_dict[33],BN)

for i in BN.keys():
    drawset.binary_tree(B.GetBDD(BN[i]),fname=str(i))
"""







