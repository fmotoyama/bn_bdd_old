# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:25:53 2020

@author: motoyama
"""
import itertools
import numpy as np
from collections import defaultdict
from typing import NamedTuple, Union


class Node(NamedTuple):
    v: int
    n0: 'Node'  # n0: Union['Node', 'Leaf']かも
    n1: 'Node'
    hash: int

class Leaf(NamedTuple):
    value: int
    hash: int


class BDD:
    """
    n,node  : Nodeオブジェクト 同じ意味のオブジェクトは1つのみ
    h,hash  : 各ノードに割り当てられたハッシュ
    """
    def __init__(self):
        self.table = dict()
        self.table_AP = defaultdict(dict)   # table_AP[op][(nA,nB)] = 演算結果  ※(nA,nB)は、順序を気にしないときはfrozenset, 気にするときはtuple
        self.leaf0 = Leaf(0,hash((0,)))
        self.leaf1 = Leaf(1,hash((1,)))
        self.table[self.leaf0.hash] = self.leaf0
        self.table[self.leaf1.hash] = self.leaf1
        
        # &
        def __and__(self_,other):
            return self.AND(self_,other)
        # |
        def __or__(self_,other):
            return self.OR(self_,other)
        # ^
        def __xor__(self_,other):
            return self.XOR(self_,other)
        # ~
        def __invert__(self_):
            return self.NOT(self_)
        Node.__and__ = __and__; Leaf.__and__ = __and__
        Node.__or__ = __or__; Leaf.__or__ = __or__
        Node.__xor__ = __xor__; Leaf.__xor__ = __xor__
        Node.__invert__ = __invert__; Leaf.__invert__ = __invert__
    
    
    def GetNode(self, v, n0, n1):
        # 削除規則
        if n0 == n1:
            return n0
        # 共有規則
        h = hash((v,n0.hash,n1.hash))
        n = Node(v, n0, n1, h)
        n_found = self.table.get(h)
        if n_found:
            return n_found
        else:
            self.table[h] = n
            return n
    
    
    def GetUnitNode(self, v):
        return self.GetNode(v, self.leaf0, self.leaf1)
    
    
    def OR(self,nA,nB):
        # 掘り進む必要がないケース
        if nA == self.leaf1 or nB == self.leaf1:
            return self.leaf1
        if nA == nB or nB == self.leaf0:
            return nA
        if nA == self.leaf0:
            return nB
        
        # 過去の計算結果を用いる
        key = hash(frozenset({nA.hash,nB.hash}))  # 順序を気にしない
        result_found = self.table_AP['OR'].get(key)
        if result_found:
            return result_found
        
        # 再帰的にシャノン展開
        vA,nA0,nA1 = nA[:3]
        vB,nB0,nB1 = nB[:3]
        if vA<vB:
            r=self.GetNode(vA,self.OR(nA0,nB),self.OR(nA1,nB))
        elif vA>vB:
            r=self.GetNode(vB,self.OR(nA,nB0),self.OR(nA,nB1))
        elif vA==vB:
            r=self.GetNode(vA,self.OR(nA0,nB0),self.OR(nA1,nB1))
        
        self.table_AP['OR'][key] = r
        return r
        
        
    def AND(self,nA,nB):
        # 掘り進む必要がないケース
        if nA == self.leaf0 or nB == self.leaf0:
            return self.leaf0
        if nA == nB or nB == self.leaf1:
            return nA
        if nA == self.leaf1:
            return nB
        
        # 過去の計算結果を用いる
        key = hash(frozenset({nA.hash,nB.hash}))  # 順序を気にしない
        result_found = self.table_AP['AND'].get(key)
        if result_found:
            return result_found
        
        # 再帰的にシャノン展開
        vA,nA0,nA1 = nA[:3]
        vB,nB0,nB1 = nB[:3]
        if vA<vB:
            r=self.GetNode(vA,self.AND(nA0,nB),self.AND(nA1,nB))
        elif vA>vB:
            r=self.GetNode(vB,self.AND(nA,nB0),self.AND(nA,nB1))
        elif vA==vB:
            r=self.GetNode(vA,self.AND(nA0,nB0),self.AND(nA1,nB1))
        
        self.table_AP['AND'][key] = r
        return r
    
        
    def XOR(self,nA,nB):
        # 掘り進む必要がないケース
        if nA == nB:
            return self.leaf0
        if nA == self.leaf0:
            return nB
        if nA == self.leaf1:
            return self.NOT(nB)
        if nB == self.leaf0:
            return nA
        if nB == self.leaf1:
            return self.NOT(nA)
        
        # 過去の計算結果を用いる
        key = hash(frozenset({nA.hash,nB.hash}))  # 順序を気にしない
        result_found = self.table_AP['XOR'].get(key)
        if result_found:
            return result_found
        
        # 再帰的にシャノン展開
        vA,nA0,nA1 = nA[:3]
        vB,nB0,nB1 = nB[:3]
        if vA<vB:
            r=self.GetNode(vA,self.XOR(nA0,nB),self.XOR(nA1,nB))
        elif vA>vB:
            r=self.GetNode(vB,self.XOR(nA,nB0),self.XOR(nA,nB1))
        elif vA==vB:
            r=self.GetNode(vA,self.XOR(nA0,nB0),self.XOR(nA1,nB1))
        
        self.table_AP['XOR'][key] = r
        return r
    
        
    def EQ(self,nA,nB):
        # 掘り進む必要がないケース
        if nA == nB:
            return self.leaf1
        if nA == self.leaf1:
            return nB
        if nA == self.leaf0:
            return self.NOT(nB)
        if nB == self.leaf1:
            return nA
        if nB == self.leaf0:
            return self.NOT(nA)
        
        # 過去の計算結果を用いる
        key = hash(frozenset({nA.hash,nB.hash}))  # 順序を気にしない
        result_found = self.table_AP['EQ'].get(key)
        if result_found:
            return result_found
        
        # 再帰的にシャノン展開
        vA,nA0,nA1 = nA[:3]
        vB,nB0,nB1 = nB[:3]
        if vA<vB:
            r=self.GetNode(vA,self.EQ(nA0,nB),self.EQ(nA1,nB))
        elif vA>vB:
            r=self.GetNode(vB,self.EQ(nA,nB0),self.EQ(nA,nB1))
        elif vA==vB:
            r=self.GetNode(vA,self.EQ(nA0,nB0),self.EQ(nA1,nB1))
        
        self.table_AP['EQ'][key] = r
        return r
    
    
    def NOT(self,n):
        if isinstance(n,Leaf):
            if n == self.leaf0:
                return self.leaf1
            else:
                return self.leaf0
        # 過去の計算結果を用いる
        result_found = self.table_AP['NOT'].get(n.hash)
        if result_found:
            return result_found
        # 再帰的に計算
        r = self.GetNode(n.v,self.NOT(n.n0),self.NOT(n.n1))
        
        self.table_AP['NOT'][n.hash] = r
        return r
        
    
            
    ####################
    
    #!!!tableの仕様変更後、未点検
    @staticmethod
    def GetBDDdict(node):
        # bddをdict型で返す
        if isinstance(node,Leaf):
            return node.value
        bdd_dict = dict()   # hash: [v,hash0,hash1]
        def scan(node):
            if isinstance(node,Leaf):
                return
            n0_hash = node.n0.value if isinstance(node.n0,Leaf) else node.n0.hash
            n1_hash = node.n1.value if isinstance(node.n1,Leaf) else node.n1.hash
            bdd_dict[node.hash] = [node.v, n0_hash, n1_hash]
            scan(node.n0)
            scan(node.n1)
        scan(node)
        return bdd_dict
    
    
    
    def AssignConst(self,node,cdict):  #cdict[v]=True/False
        """各変数に定数を代入した結果のbddを返す"""
        if isinstance(node,Leaf):
            return node
        
        const = cdict.get(node.v)
        if const is None:
            r = self.GetNode(
                node.v,
                self.AssignConst(node.n0,cdict),
                self.AssignConst(node.n1,cdict)
                )
        else:
            assert const==0 or const==1
            r = self.AssignConst(node.n1,cdict) if const else self.AssignConst(node.n0,cdict)
        return r
    
    
    @staticmethod
    def AssignConstAll(node,cdict):
        """BDDのすべての変数の値が与えられているとき、BDDの出力を求める"""
        if isinstance(node,Leaf):
            return node.value
        while isinstance(node,Node):
            node = node.n1 if cdict[node.v] else node.n0
        return node.value


    @staticmethod
    def GetV(node):
        """bddで使われている変数を列挙する"""
        V = set()
        def scan(node):
            if isinstance(node,Leaf):
                return
            V.add(node.v)
            scan(node.n0)
            scan(node.n1)
        scan(node)
        return V


    def EnumPath(self, node, shorten=False):
        """1へ到達するパス（1を出力する状態）を全て求める"""
        if isinstance(node,Leaf):
            return None,None    # ちゃんと書く余地がある
        V_ordered = sorted(self.GetV(node))
        l = len(V_ordered)
        
        def scan(node):
            # nodeから1へ到達するための入力の組み合わせを求める
            id_v = V_ordered.index(node.v)
            states = []
            for branch,node_c in enumerate([node.n0,node.n1]):
                if node_c == self.leaf1:
                    states_sub = np.full((1, l), -1, dtype = 'i1')
                    states_sub[:,id_v] = branch
                    states.append(states_sub)
                elif isinstance(node_c,Node):
                    states_sub = scan(node_c)
                    states_sub[:,id_v] = branch
                    states.append(states_sub)
            return np.concatenate(states)
        
        states = scan(node)
        
        if not shorten:
            # -1の部分を書き下す
            states2 = []
            for state in states:
                cols = np.where(state==-1)[0]
                states2_sub = np.tile(state, (2**len(cols),1))   # stateを縦に並べる
                states2_sub[:,cols] = list(itertools.product([1,0], repeat=len(cols)))
                states2.append(states2_sub)
            states2 = np.concatenate(states2)
            states = states2
        
        return states,V_ordered
    

    def EnumPath(self, node, V=None):
        """1へ到達するパス（1を出力する状態）を全て求める"""
        if V is None:
            V = sorted(self.GetV(node))
            flag = False
        else:
            flag = True     # 欠損を書き下すフラグ
        l = len(V)
        
        if node == self.leaf0:
            return np.empty((0, l), dtype = 'i1'), V
        if node == self.leaf1:
            return np.array(list(itertools.product([1,0], repeat=l)), dtype='i1'), V
        
        def scan(node):
            # nodeから1へ到達するための入力の組み合わせを求める
            id_v = V.index(node.v)
            states = []
            for branch,node_c in enumerate([node.n0,node.n1]):
                if node_c == self.leaf1:
                    states_sub = np.full((1, l), -1, dtype = 'i1')
                    states_sub[:,id_v] = branch
                    states.append(states_sub)
                elif isinstance(node_c,Node):
                    states_sub = scan(node_c)
                    states_sub[:,id_v] = branch
                    states.append(states_sub)
            return np.concatenate(states)
        
        states = scan(node)
        
        if flag:
            # -1の部分を書き下す
            states2 = []
            for state in states:
                cols = np.where(state==-1)[0]
                states2_sub = np.tile(state, (2**len(cols),1))   # stateを縦に並べる
                states2_sub[:,cols] = list(itertools.product([1,0], repeat=len(cols)))
                states2.append(states2_sub)
            states2 = np.concatenate(states2)
            states = states2
        
        return states,V
    
    
    def CountPath(self, node, V_ordered=None):
        """
        1へ到達するパスの本数を数える
        V_orderedのうちbddに登場していない変数のパターンも数える
        V_ordered=Noneのときは単純にbddのパス数を数える
        """
        if V_ordered == None:
            def scan(node):
                if isinstance(node,Leaf):
                    return node.value
                return scan(node.n0) + scan(node.n1)
            return scan(node)
            
        else:
            l = len(V_ordered)
            def scan(node):
                if isinstance(node,Leaf):
                    return node.value, l
                # 子ノードとの間で消えているノード数に応じてcountを増やす
                id_v = V_ordered.index(node.v)
                count0, id_0 = scan(node.n0)
                count0 *= 2 ** (id_0 - id_v - 1)
                count1, id_1 = scan(node.n1)
                count1 *= 2 ** (id_1 - id_v - 1)
                return count0+count1, id_v
            
            count, id_ = scan(node)
            count *= 2 ** (id_)
            return count
    
    def PickPath(self,node):
        """ランダムにパスを1本示す"""
        if isinstance(node,Leaf):
            return
        path = dict()
        while node != self.leaf1:
            edge = np.random.randint(0,2)
            if node[1+edge] == self.leaf0:
                path[node.v] = 1-edge
                node = node[1+1-edge]
            else:
                path[node.v] = edge
                node = node[1+edge]
        return path
    
    def PickPath2(self,node):
        """パスをbddの形で返す"""
        if isinstance(node,Leaf):
            return
        path_bdd = self.leaf1
        while node != self.leaf1:
            edge = np.random.randint(0,2)
            if node[1+edge] == self.leaf0:
                edge = 1-edge
            path_bdd &= ~self.GetUnitNode(node.v) if edge==0 else self.GetUnitNode(node.v)
            node = node.n0 if edge==0 else node.n1
        return path_bdd


    
if __name__ == '__main__':
    from drawset import binary_tree
    B = BDD()
    
    bn = {v:B.GetUnitNode(v) for v in range(1,6)}
    bdd1 = B.OR(B.AND(bn[1],bn[3]),bn[2])
    bdd2 = (bn[1] & bn[3]) | bn[2]
    bdd3 = B.CalcOnBDD(bdd2, bn)
    assert bdd1.hash == bdd2.hash
    assert bdd2.hash == bdd3.hash
    
    bdd_dict = B.GetBDDdict(bdd1)
    paths,V_ordered = B.EnumPath(bdd1,shorten=True)
    num_paths = B.CountPath(bdd1, V_ordered=None)
    paths2, _ = B.EnumPath(bdd1,shorten=False,check=True)
    num_paths2 = B.CountPath(bdd1, V_ordered=V_ordered)
    
    binary_tree(bdd_dict,node_name={2:'uu'})












