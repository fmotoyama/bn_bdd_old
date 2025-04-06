# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:33:43 2021

@author: f.motoyama
"""
import copy, random, csv
import numpy as np
from time import time

import BooleanNetwork, drawset, def_f
import algorithm
#制御の実験

B = BooleanNetwork.BooleanNetwork()



# データ定義
f_dict, _ = def_f.def_f('import','f_dict_72')
V_controller = (1,2,3,4,5)    # コントローラに指定するノード　これらのノードの状態は見ない
for v in V_controller:
    del f_dict[v]


# 目標状態を設定
steady_state = B.GetSS(B.GetBN(f_dict))
target_state = steady_state[0]
#target_state = np.random.randint(0,2,len(f_dict),dtype='b')
target_state = [
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,1,1,1,
    1,1,1,1,1,1,1,0,1,0,
    0,0,0,0,0,0,1,1,1,0,
    0,0,0,0,0,0,1,
    ]
#target_state = np.zeros(67)
target_dict = {v:val for val,v in zip(target_state, sorted(f_dict))}


# BDD取得時間,経路数
N = 3
data = np.empty((N,5))

with open('data.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['BDD取得時間','計算時間','総組み合わせ数','BDD上の経路数','目標状態に到達可能な初期状態数'])


for i in range(N):
    B = BooleanNetwork.BooleanNetwork()
    T = i+1
    V_ordered = list(range(6,72+1+len(V_controller)*T))
    t1 = time()
    BN = algorithm.BCN_transition(B, f_dict, V_controller, T)
    node_sat = algorithm.BN_concatenate(B, BN, target_dict)
    t2 = time()
    num_path1 = B.CountPath(node_sat, V_ordered)
    t3 = time()
    num_path2 = B.CountPath(node_sat)
    num_reachable = B.CountPath(
        algorithm.get_reachable(B, node_sat, list(f_dict)), list(f_dict)
        )
    
    #集計
    data[i] = [t2-t1, t3-t1, num_path1, num_path2, num_reachable]
    #"""
    with open('data.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([data[i,0],data[i,1],f'\'{int(data[i,2])}',f'\'{int(data[i,3])}',f'\'{int(data[i,4])}'])
    #"""
    print(i+1, num_path2)


for d in data:
    print(f'\'{int(d[4]) / (2**67)}')


drawset.binary_tree(B.GetBDDdict(node_sat))
