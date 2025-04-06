# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:26:55 2019

@author: motoyama
"""
import math,csv
from time import time
import numpy as np

from BooleanNetwork import BooleanNetwork
from def_f import def_f
from get_FVS import get_FVS
import drawset
#from QuineMcCluskey import QM2


N = 100
label = [
    '計算時間',
    '低次元化後ノード数',
    '固定点数',
    '固定点数から求まる最小ノード数',
    '元の辺数',
    '低次元化後の辺数',
    '元の最大入り次数'
    ]
with open('data.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(label)

data = np.empty((N,len(label)))


for i1 in range(N):
    #"""
    f_dict, fvs = def_f('random', 'scalefree', n=20, all_fvs=True)
    BooleanNetwork.save_f_dict(f_dict, fvs)
    """
    f_dict, fvs = def_f('import')
    #"""
    drawset.wiring_diagram2(f_dict, f'BN_{i1}')
    
    #Bインスタンスを作成して、BNをBDDの集合として扱う　（+　fvsを求める）
    B = BooleanNetwork()
    BN = B.GetBN(f_dict)
    
    f_dict_r1 = f_dict
    t1 = time()
    for i2 in range(len(BN)):
        #卒論の低次元化, 定数を代入, 式を得る
        BN_r1 = B.SimplifyBN(f_dict_r1, fvs)
        BN_r1 = B.SimplifyBN_const(BN_r1)
        f_dict_r1 = B.BN_to_f_dict(BN_r1, check = True)
        
        #修論の低次元化
        if i2 >= 0:
            BN_r2_partially, paths = B.SimplifyBN_advance(f_dict_r1)
            BN_r2 = {**BN_r1, **BN_r2_partially}
            tgt_v = list(BN_r2_partially)
        else:
            tgt_v = list(set(tgt_v) - (set(tgt_v)-set(BN_r1)))
            BN_r2_partially, paths = B.SimplifyBN_advanceB2(f_dict_r1, tgt_v)
            BN_r2 = {**BN_r1, **BN_r2_partially}
            tgt_v = list(BN_r2_partially)
        #assert ~B.is_same_SS(BN, BN_r2), 'error'
        
        BN_r1 = BN_r2
        f_dict_r1 = B.BN_to_f_dict(BN_r1, check=True)
        fvs = get_FVS(B.f_dict_to_parent(f_dict_r1), _random = False)
        
        if paths == []:
            break
        #drawset.wiring_diagram(B.f_dict_to_parent(f_dict_r1), f'BN_{i1}_{i2}')
    
    #データの集計
    """
    0:計算時間　
    1:低次元化後ノード数 
    2:固定点数 
    3:固定点数から求まる最小ノード数　
    4:元の辺数
    5:低次元化後の辺数
    6:元の最大入り次数
    """
    data_temp = np.empty(len(label))
    data_temp[0] = time()-t1
    data_temp[1] = len(BN_r1)
    _,SS = B.getSS2(BN_r1)
    data_temp[2] = len(SS)
    data_temp[3] = math.ceil(math.log(len(SS), 2)) if len(SS)!=0 else 0
    parent_num = [len(line) for line in B.BN_to_parent(BN).values()]
    data_temp[4] = sum(parent_num)
    parent_num_r1 = [len(line) for line in B.BN_to_parent(BN_r1).values()]
    data_temp[5] = sum(parent_num_r1)
    data_temp[6] = max(parent_num)
    
    with open('data.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data_temp)
    data[i1] = data_temp
    
    assert ~B.is_same_SS(BN, BN_r2), 'SS not same'
    print(f'{i1}:{i2}, ',end='')




import pandas as pd
df = pd.read_csv('data.csv', encoding="shift-jis")
summary = df.describe()





