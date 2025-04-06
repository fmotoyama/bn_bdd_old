# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:25:53 2020

@author: motoyama
"""

import copy, itertools
import numpy as np

from BDD import BDD, Node, Leaf
# GetSSが古い


class BooleanNetwork(BDD):
    def __init__(self):
        super().__init__()
        """
        f       : 論理式(list)
        f_dict  : 式をlistで表したBN(dict)
        v       : 変数名(int)
        BN      : 式をBDDで表したBN(dict)
        node    : (Nodeクラス)
        """
        
    def GetBN(self,f_dict):
        BN = {}
        for v in f_dict:
            if f_dict[v] == []:
                #BN[v] = None
                pass
            else:
                BN[v] = self.calc_f(f_dict[v])
        return BN
    
    
    def GetSS(self, BN, check=False):
        """
        BNの定常状態を求める
        式が与えられていない変数の値は何でもよい
        """
        V_ordered = sorted(BN)
        l = len(V_ordered)
        
        def scan(SS, v_root, node):
            """node以下の情報をSSに反映させる"""
            if isinstance(node, Leaf):
                # V.index(v_root)列で「この葉と逆の値」をもつ行を除き、同列(の-1)をsignに書き換える
                SS = SS[np.where(SS[:,V_ordered.index(v_root)] != 1-node.value)[0],:]
                SS[:,V_ordered.index(v_root)] = node.value
                return SS
            
            v_id = V_ordered.index(node.v) if node.v in V_ordered else None
            # 0枝
            SS2 = []
            for edge in range(2):
                if v_id is not None:
                    # v_id列において値が1-edgeの行を除き、同列(の-1)をedgeに書き換える
                    SS_temp = SS[np.where(SS[:,v_id] != 1-edge)[0],:]
                    SS_temp[:,v_id] = edge
                    # 子ノード以下をscanさせる
                    SS_temp = scan(SS_temp, v_root, node[1+edge])
                else:
                    SS_temp = scan(SS, v_root, node[1+edge])
                SS2.append(SS_temp)
            return np.concatenate(*[SS2])
        
        SS = np.full((1,l), -1, dtype = 'i1')
        for v_root,node in BN.items():
            SS = scan(SS, v_root, node)
        
        if check:
            assert not self.CheckSS(BN,SS,V_ordered), 'GetSS: SS wrong'
        return SS
    
    
    def GetSS2(self, BN, node_set=None):
        """
        node_setの表す状態のうち、BNの固定点となるものを探す
        """
        if node_set is None:
            node_set = self.leaf1
        for v,node in BN.items():
            node_set = (node_set & node & self.GetUnitNode(v)) | (node_set & ~node & ~self.GetUnitNode(v))
        return node_set
        
    
    
    def CheckSS(self, BN, SS):
        """SSがBNで固定点となっているかチェックする"""
        V_ordered = sorted(BN)
        for state in SS:
            cdict = {v:value for v,value in zip(V_ordered,state)}
            for i,v in enumerate(V_ordered):
                output = self.AssignConst(BN[v],cdict)
                if isinstance(output,Leaf):
                    if state[i] != output.value:
                        return state    
                if set(V_ordered) & self.GetV(output):
                    return state
        return 0
    
    
    
    def BN_to_f_dict(self, BN, check = False):
        """
        BN(BDDの集合)からf_dictを求める
        check == Trueのとき、求めたf_dictからBNを作成し、元と変わらないか確認する
        """
        f_dict = {v:self.BDD_to_f(node) for v,node in BN.items()}
        if check:
            BN2 = self.GetBN(f_dict)
            for v in BN:
                assert (BN[v] == BN2[v]), f'BN_to_f_dict: failed v={v}'
        return f_dict
    
    
    def BN_to_parent(self,BN):
        """BNからparentを返す"""
        return {v:self.GetV(node) for v,node in BN.items()}
        
    
    def f_dict_to_parent(self,f_dict):
        """f_dictからparentを返す"""
        return {
            v: list(set(map(abs,itertools.chain.from_iterable(f))))
            for v,f in f_dict.items()
            }
    
    
    def nextf(self, BN, f_dict, n = 1):
        """BNに対してf_dictの演算をn回行う"""
        BN2 = copy.copy(BN)
        for i in range(n):
            BN_temp = {v:self.calc_f(f_dict[v], BN2) for v in BN2}
            BN2 = copy.copy(BN_temp)
        return BN2
    
    
    def SimplifyBN_const(self, BN):
        """定数となっている関数を代入する"""
        while True:
            #定数である関数を探し、clistを完成させる
            cdict = {}
            for v,node in BN.items():
                if isinstance(node,Leaf):
                    cdict[v] = node.value
            #clistが空のとき、終了
            if cdict == {}:
                return BN
            #clistを全関数に代入
            BN2 = {}
            for v in BN:
                if isinstance(BN[v],Node):
                    BN2[v] = self.AssignConst(BN[v],cdict)
            BN = BN2
    
    
    def BDD_to_f(self, node):
        """
        BDDを見て、1に到達する状態の組み合わせから積和標準形のfを得る
        fは最小とは限らない
        """
        if isinstance(node,Leaf):
            return node.value
        f = []
        states,V = self.EnumPath(node, shorten=True)
        for state in states:
            term = [v * [-1,1][bool(value)] for v,value in zip(V,state) if value != -1]
            f.append(term)
        return f
    
    
    def BN_to_TT(self, BN):
        V_ordered = sorted(BN)
        l = len(V_ordered)
        TT = np.empty((2, 2**l, l), dtype='i1')
        
        # 真理値表の左側
        TT[0] = list(itertools.product([1,0], repeat=l))
        # 真理値表の右側
        for row,state in enumerate(TT[0]):
            cdict = {var:val for var,val in zip(V_ordered,state)}
            TT[1,row] = [self.AssignConstAll(BN[v],cdict) for v in V_ordered]
        return TT
    
    
    def TT_to_f_dict(self, TT, V_ordered=None):
        """TT : (2, 2**l, l)"""
        f_dict = {}
        l = TT.shape[2]
        if not V_ordered:
            V_ordered = np.arange(1,l+1)
        for col,v in enumerate(V_ordered):
            rows = np.where(TT[1,:,col])[0]
            f = []
            for row in rows:
                term = [V_ordered[col2]*(-1,1)[v2] for col2,v2 in enumerate(TT[0,row])]
                f.append(term)
            f_dict[v] = f
        return f_dict
    
    
    def f_dict_to_TT(self,f_dict):
        """fから直接計算する"""
        V = list(f_dict)
        l = len(V)
        TT = np.empty((2, 2**l, l), dtype='i1')
        # 真理値表の左側
        TT[0] = list(itertools.product([1,0], repeat=l))
        # 真理値表の右側
        for col in range(l):
            TT[1,:,col] = self.calc_on_f(f_dict[V[col]],TT[0],V)
        return TT
    
    
    def calc_f(self, f:list, bdds:dict=dict()) -> Node:
        """BDDについて関数fの演算を行う"""
        # fが空のとき
        if f == []:
            return None
        # fが定数のとき
        if isinstance(f,int):
            return f
        
        def calc_product(term:list):
            # termがもつ変数を全て積算する
            assert len(term)
            node = self.leaf1
            for v in term:
                node2 = bdds.get(abs(v))
                if not node2:
                    # BNにない変数を使おうとする場合、その変数を表すBDDを自動で用意する
                    node2 = self.GetUnitNode(abs(v))
                if v<0:
                    node2 = self.NOT(node2)
                node = self.AND(node, node2)
            return node
        
        #先頭のtermのBDDを生成
        node_sum = calc_product(f[0])
        #termごとにBDDを生成し、sum_BDDに和算する
        for term in f[1:]:
            node_sum = self.OR(node_sum, calc_product(term))
        return node_sum
    
    
    @staticmethod
    def calc_on_f(f, states, V):
        """fをそのまま用いて計算する"""
        output = []
        for state in states:
            y = 0
            for term in f:
                y_term = 1
                for x in term:
                    val = state[V.index(abs(x))]
                    val = not val if x < 0 else val
                    y_term = y_term and val
                y = y or y_term
            output.append(y)
        return np.array(output, dtype='i1')
    
    
    @staticmethod
    def is_equal_BN(BN1,BN2):
        for (v1,node1),(v2,node2) in zip(BN1.items(),BN2.items()):
            assert v1 == v2
            assert node1 == node2
        return 0
    
    @staticmethod
    def save_f_dict(f_dict, name = 'f_dict', **kwargs):
        """f_dictをtxtファイルとして出力する"""
        #shutil.rmtree('f_dict.txt')
        with open(f'{name}.txt', mode='w') as f:
            f.write(str(f_dict))
            if kwargs:
                f.write('\n')
                f.write(str(kwargs))

if __name__ == '__main__':
    import drawset, def_f
    from QuineMcCluskey import QM
    B = BooleanNetwork()
    
    f_dict,_ = def_f.def_f('import','f_dict_15')
    BN = B.GetBN(f_dict)
    #drawset.wiring_diagram(B.f_dict_to_parent(f_dict), 'wd')
    #drawset.transition_diagram(B.BN_to_TT(BN),'td')
    
    for i in range(10000):
        f_dict,_ = def_f.def_f('random','normal',n=10)
        #f_dict,_ = def_f.def_f('import','f_dict')
        BN = B.GetBN(f_dict)
        SS1 = B.GetSS(BN)
        if len(SS1) != B.CountPath(B.GetSS2(BN), list(BN)):
            B.save_f_dict(f_dict)
            print('error')
            break
        print(f'\r{i}', end='')
    
    
    
    

