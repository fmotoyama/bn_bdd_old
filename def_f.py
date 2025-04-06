# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:42:14 2020

@author: f.motoyama
"""
import random
import numpy as np
import itertools


def def_f(mode, *args, **kwargs):
    """
    f_dictを定義する
    input : (mode, *args, **kwargs) =
        ('manual') or
        ('random', (network->str), {all_fvs->bool}) or
        ('import', (file_name->str)')
    """
    f_dict = {}
    
    if mode == "manual":
        f_dict = {1:[[1]],2:[[1,-3]],3:[[-2,4]],4:[[1],[3]]}
        info = None
        
        
    elif mode == "random":
        #ネットワーク生成
        #normal(prob)/n_scalefree,n_scalefree2(gamma)/sf_BA(m)
        temp = {
            'n':20,
            'prob_not':0.3,
            'all_fvs':False,
            'prob':0.15,
            'gamma':3.0     # gammaを大きくすると疎になる
            }
        kwargs = {**temp, **kwargs}
        
        n = kwargs['n']
        prob_not = kwargs['prob_not']
        if args[0] == "normal":
            pnode = get_network('normal', n, prob=kwargs['prob'])
        if args[0] == "scalefree":
            pnode = get_network('scalefree', n, gamma=kwargs['gamma'])
        
        #ネットワークから式を生成
        #AND/OR/ANDOR/ANDOR2
        f_dict = get_f_dict(pnode, 'ANDOR2', prob_not)
        
        info = None
        if kwargs['all_fvs']:
            for x,f in f_dict.items():
                V = list(itertools.chain.from_iterable(f))
                V = list(set([v if 0 < v else v*(-1) for v in V]))
                if x not in V:
                    f.append([x])
            info = {'fvs': list(f_dict)}
    
    
    elif mode == 'import':
        #同じディレクトリにあるfile_nameからf_dictを得る
        name = f'data/{args[0]}.txt' if args else 'data/f_dict.txt'
        with open(name, mode='r') as f:
            l = f.readlines()
            f_dict = eval(l[0])
            info = eval(l[1]) if len(l) == 2 else None
        return f_dict,info
    
    return f_dict, info


def get_network(nw, n, **kwargs):
    #ネットワークの生成
    if nw == 'normal':
        pnode = make_n_normal(n, kwargs['prob'])
    elif nw == 'scalefree':
        pnode = make_n_scalefree(n, kwargs['gamma'])
    elif nw == 'scalefree_BA':
        pnode = make_n_scalefree_BA(n, kwargs['m'])
    return pnode


def make_n_normal(n, prob):
    """probの確率で2頂点間の有向辺が結ばれる 最低1つの入り次数をもつこととする"""
    v_list = list(range(1,n+1))
    pnode = {}
    for v in v_list:
        f = [v for v in v_list if random.random()<=prob]
        if f == []:
            f = [random.choice(v_list)]
        pnode[v] = f
    return pnode


def make_n_scalefree(n, gamma):
    """"
    スケールフリーネットワークっぽく、各ノードの親ノードをランダムに決定してpnodeを返す
    n                  :ノード数
    p(k) = 1/k^gamma   :ノードが次数kをもつ確率の指数　gammaを大きくすると疎になる
    """
    #ノード番号の集合
    node = range(1, 1+n)
    #各ノードの親ノードを記録
    pnode = {}
    #各ノードの次数を記録
    #dim_list = np.zeros(n)
    
    #p_list[i] = (入り次数がiより下になる確率 / p_sum)　次数0になる確率は0
    p_list = np.empty([n+2])
    p_list[0] = 0; p_list[1] = 0
    for i in range(2,n+2):
        p_list[i] = p_list[i-1] + 1/(i**gamma)
    p_sum = p_list[-1]
    
    #ノードxの入り次数と親ノードの決定　plist[i-1]<prob1<plist[i](prob2)のとき、次数はi-1
    for i,x in enumerate(node):
        prob1 = random.random() * p_sum          #iの入り次数の確率
        for dim, prob2 in enumerate(p_list):
            """
            #入り次数が、全ノード数の半数を超えないようにする
            if 10 < dim:
                pnode[x] = random.sample(node, dim-1)    #iの親として、nodeからdim-1個を重複なく選ぶ
                pnode[x].sort()
                break
            #"""
            if prob1 < prob2:
                #dim_list[i] = dim                       #1<=d<=nの値が来うる
                pnode[x] = random.sample(node, dim-1)    #iの親として、nodeからdim-1個を重複なく選ぶ
                pnode[x].sort()
                break
    
    return pnode


def make_n_scalefree2(n, gamma):
    """"
    入り次数0を許す　!!!!!接続しないノードの処理が問題!!!!!
    """
    #ノード番号の集合
    node = range(1, 1+n)
    #各ノードの親ノードを記録
    pnode = {}
    
    #p_list[i] = (入り次数がiより下になる確率 / p_sum)　次数0になる確率は0
    p_list = np.empty([n+2])
    p_list[0] = 0
    for i in range(1,n+2):
        p_list[i] = p_list[i-1] + 1/(i**gamma)
    p_sum = p_list[-1]
    
    #ノードxの入り次数と親ノードの決定
    for i,x in enumerate(node):
        prob1 = random.random() * p_sum          #iの入り次数の確率
        for dim, prob2 in enumerate(p_list):
            if prob1 < prob2:
                pnode[x] = random.sample(node, dim-1)    #iの親として、nodeからdim-1個を重複なく選ぶ
                pnode[x].sort()
                break
    
    return pnode


def make_n_scalefree_BA(n, m = 3):
    """"
    Barabasi_Albertモデル　自己ループが存在しない
    """
    #無向グラフを隣接行列で表す
    G = np.zeros((n,n), dtype = 'bool')
    #各ノードの入り次数
    prob_list = np.zeros(n, dtype = 'int8')
    #初期値として、mノードの完全グラフを作成
    prob_list[:m] = m-1
    cmbs = list(itertools.combinations(range(m),2))
    for cmb in cmbs:
        G[cmb[0],cmb[1]] = True; G[cmb[1],cmb[0]] = True
    
    #BAモデルの作成　m+1個目のノードを置くところから開始
    for x in range(m,n):
        #ノードを追加するときm本の辺を引く
        prob = prob_list[:x]    #存在するノードごとの確率分布
        prob_sum = np.sum(prob)
        prob = prob / prob_sum
        #確率分布probで、xの親をm個選ぶ
        x2s = np.random.choice(a = range(x), size = m, p = prob, replace=False)
        #接続情報の更新
        for x2 in x2s:
            G[x2,x] = True; G[x,x2] = True
            prob_list[x2] += 1; prob_list[x] += 1
    
    """
    #無向グラフの隣接行列を、方向がランダムな有向グラフの隣接行列に変換
    #対角線を除いた上半分のみを調べればよいので、一番下の行は調べなくてよい
    for x in range(n-1):
        idxs = np.where(G[x,x+1:])[0] + x+1
        for x2 in idxs:
            d = np.random.randint(2)
            if d == 0:
                G[x,x2] = False
            elif d == 1:
                G[x2,x] = False
    """
    #番号が小さい方を親とする   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ループが発生しない
    for x in range(n):
        G[x,:x] = False
    
    #隣接行列を隣接リストに変換 ノード番号は1から
    pnode = {}
    for x in range(n):
        pnode[x+1] = (np.where(G[x])[0] + 1).tolist()
    
    return pnode

    


def get_f_dict(pnode, operator, prob_not = None):
    """
    ネットワーク（親子関係）を作成し、論理式を作成する
    operator    :fdictで使う演算子(AND/OR/ANDOR) ANDORでは完全ランダムな論理式が与えられる
    prob_not    :引数をnot化させる確率
    """
    #not化
    if prob_not != None:
        for f in pnode:
            pnode[f] = [v * [1,-1][prob_not > random.random()] for v in pnode[f]]
    
    #ネットワークに関数を与える
    f_dict = {}
    if operator == 'AND':
        for key in pnode:
            f_dict[key] = [pnode[key]]
    elif operator == 'OR':
        for key in pnode:
            f_dict[key] = [[v] for v in pnode[key]]
    
    elif operator == 'ANDOR1':
        #ランダムな真理値表を生成しQM法で論理式にする
        pass
        
    elif operator == 'ANDOR2':
        #使用する変数をシャッフルし適当な演算子でつなげる prob_notに従って否定化
        for key in pnode:
            V = pnode[key]
            #ノードkeyの入り次数がない場合、関数は空
            if V == []:
                f_dict[key] = []
                continue
            #ランダムな論理式を作成
            V_shuffle = random.sample(V, len(V))
            f = [[V_shuffle[0]]]    #最初の変数を配置
            for v in V_shuffle[1:]:
                if random.random() < 0.5:
                    f.append([v])
                else:
                    f[-1].append(v)
            
            f_dict[key] = f
        
    return f_dict
    




if __name__ == '__main__':
    #pnode = make_n_scalefree_BA(100, prob_not = None)
    #import drawset
    #drawset.wiring_diagram(pnode,fname='wiring_diagram')
    for _ in range(100):
        f_dict, fvs = def_f('random', 'all_fvs')












