# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:19:30 2022

@author: motoyama
"""
from time import time
import numpy as np
import BooleanNetwork, drawset, def_f
from QuineMcCluskey import QM2
from BDD import Root, Node, Leaf





def is_equal_f_dict_and_BN(f_dict):
    """
    BNがf_dictを表すことを確認する
    BN → f_dict2 → BN2
    BN → TT → f_dict3 → BN3
    """
    BN = B.GetBN(f_dict)
    
    f_dict2 = B.BN_to_f_dict(BN)
    BN2 = B.GetBN(f_dict2)
    B.is_equal_BN(BN,BN2)
    
    TT = B.BN_to_TT(BN)
    f_dict3 = B.TT_to_f_dict(TT)
    BN3 = B.GetBN(f_dict3)
    B.is_equal_BN(BN,BN3)


def is_correct_calc(BN):
    """
    式を用いた計算と手動の計算が一致することを確認
    """
    bdd_1 = B.calc_f([[1,-2], [-3,4]],BN)
    
    bdd2 = B.FlipNode(BN[2])
    bdd3 = B.FlipNode(BN[3])
    bdd_2 = B.AP(
        'OR',
        B.AP('AND',bdd3,BN[4]),
        B.AP('AND',BN[1],bdd2)
        )
    assert (bdd_1.neg == bdd_2.neg) and (bdd_1.node == bdd_2.node)


def is_same_SS(BN1, BN2):
    """SS1,SS2の個数が同じこと、SSが正しいことを確認する"""
    SS1 = B.GetSS(BN1)
    SS2 = B.GetSS(BN2)
    assert not B.CheckSS(BN1, SS1), 'is_same_SS: wrong SS1'
    assert not B.CheckSS(BN2, SS2), 'is_same_SS: wrong SS2'
    assert len(SS1) == len(SS2), 'is_same_SS: diferent SS'


def is_correct_path(f_dict):
    """
    f_dict → BN → path_num
    f_dict → path_num
    """
    BN = B.GetBN(f_dict)
    t1 = time()
    path_num1 = {v:B.CountPath(root, list(f_dict)) for v,root in BN.items()}
    t1 = time() - t1
    
    V = list(f_dict)
    l = len(f_dict)
    TTl = np.empty((2**l, l), dtype='i1')
    for col in range(l):
        TTl[:,col] = ([0]*2**col + [1]*2**col) * 2**(l-col-1)
    t2 = time()
    path_num2 = {v:np.count_nonzero(B.calc_on_f(f, TTl, V)) for v,f in f_dict.items()}
    t2 = time() - t2
    
    for v in V:
        assert path_num1[v] == path_num2[v]
    return t1,t2



N = 1000
t = [0,0]
for i in range(N):
    B = BooleanNetwork.BooleanNetwork()
    f_dict, fvs = def_f.def_f('random','normal', n=10, prob_not=0.5)
    #f_dict, fvs = def_f.def_f('import','f_dict')
    B.save_f_dict(f_dict)    
    BN = B.GetBN(f_dict)
    BN_bdd = {v:B.GetBDD(root) for v,root in BN.items()}
    
    t1,t2 = is_correct_path(f_dict)
    t[0] += t1
    t[1] += t2
    
    print("\r"+str(i), end='')






