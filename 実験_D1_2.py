# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:33:43 2021

@author: f.motoyama
"""
import copy, random, csv
import numpy as np
from time import time

from BDD import Node, Leaf
import BooleanNetwork, drawset, def_f
from QuineMcCluskey import QM
import algorithm

#制御の実験
B = BooleanNetwork.BooleanNetwork()


# f_dict, V_controller, target_state, target_dact を定義

'''
# アポトーシスネットワーク
f_dict, _ = def_f.def_f('import','f_dict_72')
V_controller = (1,2,3,4,5)    # コントローラに指定するノード　これらのノードの状態は見ない
for v in V_controller:
    del f_dict[v]
# 目標状態を設定
target_state = np.array([
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,1,1,1,
    1,1,1,1,1,1,1,0,1,0,
    0,0,0,0,0,0,1,1,1,0,
    0,0,0,0,0,0,1,
    ])
#target_state = np.random.randint(0,2,len(f_dict),dtype='b')
target_dict = {v:val for val,v in zip(target_state, sorted(f_dict))}
T = 10
'''

# ランダムネットワーク
T = 2
"""
f_dict,_ = def_f.def_f('random','scalefree', n=20, gamma=3)
#f_dict, V_controller = add_pinning_node(f_dict)
V_controller = (1,2,3,4,5,6,7)
for n in V_controller:
    del f_dict[n]
B.save_f_dict(f_dict, V_controller=V_controller)
"""
f_dict = {1:[[1,3,4],[2]],2:[[3],[4]],3:[[-1]]}
V_controller = (4,)
#"""
target_state = np.array([True,True,False], dtype=np.bool_)
target_dict = {v:val for val,v in zip(target_state, sorted(f_dict))}
#'''

# bdd上の制御ノードの命名規則を整理
controller_info = np.empty((len(V_controller)*T,3), dtype='i1')
controller_info[:,0] = np.arange(max(f_dict)+1, max(f_dict)+1+len(V_controller)*T)  # bdd上の変数名
controller_info[:,1] = np.tile(np.arange(1, 1+len(V_controller)), T)                # 制御ノードとしての元の変数名
controller_info[:,2] = [i//len(V_controller) for i in range(len(V_controller)*T)]   # 時刻

# BDDの時間遷移・合成
BN_T = algorithm.BCN_transition(B, f_dict, V_controller, T)
node_sat = algorithm.BN_concatenate(B, BN_T, target_dict)
f_node_sat = B.BDD_to_f(node_sat)
# 描画
name_node = {
    int(v): f'u{v_u}({t})'
    for v,v_u,t in controller_info
    }
for v in f_dict:
    name_node[v] = f'x{v}(0)'
drawset.binary_tree(B.GetBDDdict(node_sat),f'BDD_conv(t+{T})',name_node)
paths = B.EnumPath(node_sat, shorten=True)[0]

# 経路を1本抽出 初期状態と制御入力列を決定
path = B.PickPath(node_sat)
#path = {122: 0, 121: 0, 120: 0, 119: 0, 117: 0, 116: 0, 115: 0, 114: 0, 112: 0, 111: 0, 110: 0, 109: 0, 107: 0, 106: 0, 105: 0, 104: 0, 102: 0, 101: 0, 100: 0, 99: 0, 97: 0, 96: 0, 95: 0, 94: 0, 92: 0, 91: 0, 90: 0, 89: 0, 87: 0, 86: 0, 85: 0, 84: 0, 82: 0, 81: 0, 80: 0, 79: 0, 76: 0, 75: 0, 74: 0, 69: 1, 68: 0, 67: 0, 66: 0, 64: 0, 63: 0, 62: 1, 61: 1, 59: 0, 55: 1, 49: 1, 48: 1, 47: 0, 46: 1, 45: 0, 44: 0, 43: 1, 41: 0, 40: 0, 39: 1, 38: 0, 37: 0, 36: 0, 35: 0, 34: 0, 33: 0, 32: 0, 31: 0, 30: 0, 29: 0, 28: 0, 26: 1, 25: 0, 24: 0, 23: 0, 22: 1, 20: 0, 19: 0, 18: 0, 17: 0, 16: 0, 14: 0, 13: 0, 12: 0, 10: 0, 9: 0, 8: 1, 7: 0, 6: 0}
initial_state = {v:val for v,val in path.items() if v in f_dict}
control_sequence = {v:val for v,val in path.items() if v not in f_dict}
#initial_state = {v:np.random.randint(0,2) for v in f_dict}
#control_sequence = {v:np.random.randint(0,2) for v in range(73,123)}


# 初期状態と制御入力列を用いて状態遷移
BNs_assigned = algorithm.BN_control(B, f_dict, V_controller, T, initial_state, control_sequence)
BNs_assigned_bdd = {t:{v:B.GetBDDdict(node) for v,node in BN_assigned.items()} for t,BN_assigned in BNs_assigned.items()}

# ハミング距離の推移を求める
V = list(f_dict)
states = np.full((T+1,len(V)), -1, dtype='i1')
for i,BN_assigned in enumerate(BNs_assigned_bdd.values()):
    for v,val in BN_assigned.items():
        if isinstance(val,int):
            states[i,V.index(v)] = val
hamming_distance = []
for state in states:
    idx = np.where(state!=-1)[0]
    hamming_distance.append(np.count_nonzero(state[idx] != target_state[idx]))
assert hamming_distance[-1] == 0




"""
# 全ての初期状態で、目標状態に到達できる制御入力の組み合わせの個数を求める
count_max = 0
initial_state_max = None
initial_state = {}
def scan(node, initial_state):
    global count_max, initial_state_max
    if isinstance(node, Leaf):
        return
    initial_state = copy.copy(initial_state)
    if node.v in controller_info[:,0]:
        node_sat_u = B.AssignConst(node_sat, initial_state)
        count = B.CountPath(node_sat_u)
        if count_max < count:
            count_max = count
            initial_state_max = initial_state
    else:
        initial_state[node.v] = 0
        scan(node.n0, initial_state)
        initial_state[node.v] = 1
        scan(node.n1, initial_state)
scan(node_sat,initial_state)
print(count_max)
#"""

#"""
# 初期状態を指定して制御
# 27 initial_state_maxとして求めたもの
initial_state = {6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 22: 1, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 43: 1, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 1, 55: 0, 56: 0, 59: 0, 60: 0, 61: 1, 62: 1, 63: 1, 64: 1, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0}
#initial_state = initial_state_max
node_sat_u = B.AssignConst(node_sat, initial_state)
count = B.CountPath(node_sat_u)
drawset.binary_tree(B.GetBDDdict(node_sat_u),f'BDD_conv(t+{T})_u',name_node)
c1 = algorithm.control1(B, node_sat_u, V_controller, controller_info)

#"""


"""
# 目標状態に到達可能な初期状態の集合
node_reachable = algorithm.get_reachable(B, node_sat, controller_info)
#drawset.binary_tree(B.GetBDDdict(node_reachable),f'BDD_conv(t+{T})_0')
#count = B.CountPath(node_reachable, V_ordered=sorted(f_dict))
#count = B.CountPath(node_reachable)
paths,V_ordered = B.EnumPath(node_reachable, shorten=False)
#"""


"""
# 初期状態、制御入力列の文字起こし
control_sequence2 = []
for t in range(10):
    c2 = []
    for i in range(5):
        index = 72+1 + i + t*5
        if index not in control_sequence:
            c2.append('*')
        else:
            c2.append(control_sequence[index])
    control_sequence2.append(c2)
initial_state2 = []
for v in range(6,73):
    index = 72+1 + i + t*5
    if v not in initial_state:
        initial_state2.append('*')
    else:
        initial_state2.append(initial_state[v])
#"""











