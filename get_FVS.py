# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:16:26 2020

@author: motoyama
"""
import random
from copy import copy,deepcopy

def get_FVS(parent, _random = True):
    """parentの示すインタラクショングラフについてFVSを求める"""
    fvs = []

    def delete_v(vlist):
        """cpからvlistのノードを消去する"""
        will_delete = []
        
        for v in vlist:
            #削除ノードの子から、親の情報を削除
            for vc in cp[v][0]:
                #vの子ノードごと
                cp[vc][1].remove(v)
                if cp[vc][1] == [] and vc not in vlist:
                    will_delete.append(vc)
            #削除ノードの親から、子の情報を削除
            for vp in cp[v][1]:
                cp[vp][0].remove(v)
                if cp[vp][0] == [] and vp not in vlist:
                    will_delete.append(vp)
            #削除ノードを削除
            del cp[v]
        
        #親のいなくなったノードを再帰的に削除
        will_delete = list(set(will_delete))
        if will_delete != []:
            delete_v(will_delete)
    
    def scan():
        """parentをてきとうに進みtrailを記録　ループに至ったとき、trailからループを抽出して返す"""
        #スタートするノードをランダムに抽出
        if _random == True:
            vst = random.choice(list(cp))
        else:
            vst = list(cp)[0]
        trail = [vst]
        
        for i in range(l):
            #行先をシャッフル
            if _random == True:
                vp = random.choice(cp[vst][1])
            else:
                vp = cp[vst][1][0]
            #ループを検知
            if vp in trail:
                start = trail.index(vp)
                return trail[start:]
            else:
                trail.append(vp)
                vst = vp
    
    
    #cpに各ノードの子・親を格納
    cp = deepcopy(parent)           #[child,parent,次数]
    for k in cp:
        cp[k] = [[], cp[k], len(cp[k])]
    for k,value in cp.items():
        for v in value[1]:
            cp[v][0].append(k)
            cp[v][2] += 1
    #自己ループのある or 親または子がいないノードを削除
    del_node = []
    for k,v in cp.items():
        if k in v[1]:
            fvs.append(k)
            del_node.append(k)
        elif v[0] == [] or v[1] == []:
            del_node.append(k)
    delete_v(del_node)
    

    l = len(cp)
    for i in range(l):
        #ループを求める
        loop = scan()
        #loopで最も次数の大きいノードをfvsとする
        fvsv = loop[0]
        for v in loop:
            if cp[fvsv][2] < cp[v][2]:
                fvsv = v
        fvs.append(fvsv)
        delete_v([fvsv])
        if len(cp) == 0:
            break

    return sorted(fvs)




#parent = {0: [32, 7, 14, 17, 25, 31], 1: [7], 2: [11], 3: [35], 4: [35], 5: [2, 43, 12, 46, 15, 47, 19, 30], 6: [0, 2, 3, 4, 6, 7, 8, 9, 11, 14, 17, 18, 22, 23, 25, 26, 27, 28, 29, 32, 33, 34, 36, 38, 39, 46, 49], 7: [16, 44, 29], 8: [32, 2, 3, 4, 38, 40, 41, 8, 44, 45, 47, 15, 49, 21, 25, 26, 28], 9: [2, 37], 10: [36, 5, 37, 39, 8, 43, 12], 11: [1, 4, 5, 6, 8, 9, 10, 13, 14, 15, 18, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 42, 45], 12: [49, 28], 13: [4, 7], 14: [6], 15: [40, 25, 20, 29], 16: [9], 17: [13], 18: [0, 12, 46, 31], 19: [9], 20: [26, 29], 21: [44], 22: [29], 23: [22], 24: [35], 25: [9], 26: [34], 27: [8], 28: [41], 29: [1], 30: [24], 31: [47], 32: [40], 33: [23], 34: [45], 35: [34], 36: [37], 37: [0, 19, 20, 46], 38: [33, 46], 39: [30], 40: [48, 9, 3], 41: [8], 42: [5], 43: [24, 49, 2, 25], 44: [44, 29], 45: [17], 46: [11], 47: [25, 7], 48: [6], 49: [24]}
#FVS = get_FVS(parent)











