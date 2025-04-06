# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:26:55 2019

@author: motoyama
"""

import BooleanNetwork
from def_f import def_f
import drawset
from QuineMcCluskey import QM2

N = 1
for i in range(N):
    # 関数を定義
    #f_dict, fvs = def_f('import','f_dict')
    f_dict, fvs = def_f('random','normal', n=10, prob=0.9, prob_not=0.5)
    #f_dict = {1:[[3]],2:[[3]],3:[[3]]}
    
    # BNをBDDの集合として扱う
    B = BooleanNetwork.BooleanNetwork()
    B.save_f_dict(f_dict) 
    BN = B.GetBN(f_dict)
    B.save_f_dict(f_dict)
    
    paths = [B.PickPath(BN[1]) for _ in range(5)]
    #真理値表を得る
    #TT = B.MakeTT(BN)
    
    #BNの定常状態を求める
    SS = B.GetSS(BN)
    wrong = B.CheckSS(BN, SS)
    assert not wrong
    
    #BNの式を得る
    f_dict2 = B.BN_to_f_dict(BN, check=True)
    
    #BN,r_BNのインタラクショングラフを描画
    #drawset.wiring_diagram(B.BN_to_parent(BN),"BN")
    
    #if i % (N // 4) == 0:
    #    print(i)

    f = [[1,3,4,5],[-1,2,3],[-1,2,4],[-1,2,5],[2,3,5],[2,4,5]]
    path,_ = B.EnumPath(B.calc_f(f,dict()), shorten=True)









