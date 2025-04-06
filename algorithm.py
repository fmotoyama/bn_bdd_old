# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:31:35 2023

@author: fmotoyama
"""
import copy, itertools
import numpy as np
from collections import defaultdict
from typing import NamedTuple

from BDD import Leaf
from QuineMcCluskey import QM


def ReduceBN(B, f_dict, fvs):
    """fvs[]を用いて低次元化ネットワークを求める"""
    parent = B.f_dict_to_parent(f_dict)
    
    def get_order(v):
        """親ノードからvへ到達するorderを完成させる"""
        for v_p in parent[v]:
            if v_p in fvs or v_p in order:
                # v2がfvsのとき or v2に訪ねたことがあるとき（v2の親の探索が終わっているとき）
                continue
            # v2に初めて訪ねたので、v2の親の探索をする
            get_order(v_p)
        order.append(v)
    
    # orderに従って代入
    r_BN = {}
    orders = {}
    BN_sub_org = {v: B.GetUnitNode(v) for v in f_dict}
    for v_f in fvs:
        order = []
        get_order(v_f)
        orders[v_f] = order
        #orderに従って演算
        BN_sub = copy.copy(BN_sub_org)
        for v in order:
            BN_sub[v] = B.calc_f(f_dict[v],BN_sub)
        r_BN[v_f] = BN_sub[v_f]
            
    return r_BN, orders


def GetBasin(TT, state):
    """stateのbasinを求める"""
    R = [state]
    target = [state]
    while target:
        target2 = []
        for state in target:
            states_p = TT[0,np.all(TT[1] == state, axis=1)]
            states_p = states_p[np.any(states_p != state, axis=1)]  # ループを削除
            states_p = [state for state in states_p]
            R.extend(states_p)
            target2.extend(states_p)
        target = target2
    return np.vstack(R)


def GetPinningNode(TT, SS, state_target):
    """すべての定常状態をbasinへ飛ばす最小ピニングノードを総当たりで求める"""
    # state_targetは定常状態であること
    assert np.any(np.all(SS == state_target, axis=1))
    SS = SS[np.any(SS != state_target, axis=1)]
    
    l = len(state_target)
    basin = GetBasin(TT, state_target)
    
    R = []  # 同じ長さで複数の解が存在することが考えられる
    for num in range(1,l):
        # num: ピニングノード数
        for nodes in itertools.combinations(range(l), l-num):
            # nodes: ピニングノードでないノード
            for ss_sub in SS[:,nodes]:
                if ~np.any(np.all(basin[:,nodes] == ss_sub, axis=1)):
                    break
            else:
                # すべての定常状態が、このピニングノードでbasinのどこかへ行けるとき
                R.append(sorted(set(range(l)) - set(nodes)))
        if R:
            break
    if R == []:
        R = [list(range(l))]
    return R
                
        
#-----D1,2-----
def BCN_transition(B, f_dict:dict, V_controller:list, T:int):
    """
    コントローラである変数の変数名を毎時刻変更しながら状態遷移を行う
    """
    f_dict_sub = copy.deepcopy(f_dict)
    BN = {v:B.GetUnitNode(v) for v in f_dict_sub}
    v_last = max(f_dict_sub)    # 現在の最後尾の変数名
    
    for t in range(1, T+1):
        # コントローラである変数に、新たな変数名を与える
        # 時間順 → 変数名順
        V_converter = {v:v_last+1+len(V_controller)*(t-1)+i for i,v in enumerate(V_controller)}
        f_dict_temp = dict()
        for v,f in f_dict_sub.items():
            f_temp = []
            for term in f:
                term_temp = [V_converter[abs(v)] * [-1,1][0<v] if abs(v) in V_controller else v for v in term]
                f_temp.append(term_temp)
            f_dict_temp[v] = f_temp
        
        # 時間を1進める
        BN = B.nextf(BN, f_dict_temp, n=1)
        BDDdicts = {v:B.GetBDDdict(node) for v,node in BN.items()}
    
    return BN


def BN_concatenate(B, BN, target_dict):
    """BNが目標状態に到達するための条件を表すBDDを得る"""
    bdd_sat = B.leaf1
    for v,bdd in BN.items():
        temp = bdd if target_dict[v] else B.NOT(bdd)
        bdd_sat = B.AND(bdd_sat, temp)
    return bdd_sat


def get_reachable(B, node_sat, V_initial):
    """
    目標状態に到達可能な初期状態を表すbddを得る
    V_initial: 初期状態と対応するラベル
    """
    def scan(node):
        if isinstance(node,Leaf):
            return node
        if node.v not in V_initial:
            return B.leaf1
        return B.GetNode(node.v, scan(node.n0), scan(node.n1))
    return scan(node_sat)
        


def add_pinning_node(B, f_dict):
    # ピニング制御のための制御ノードを付与する
    BN = B.GetBN(f_dict)
    V_controller = tuple(range(max(BN)+1,max(BN)+1+len(BN)))
    nodes_controller = [B.GetUnitNode(v) for v in V_controller]
    BN = {
        v: B.OR(B.AND(B.NOT(node),B.NOT(node_c)),B.AND(node,node_c))
        for (v,node),node_c in zip(BN.items(),nodes_controller)
        }
    f_dict2 = B.BN_to_f_dict(BN)
    f_dict2 = {v: QM(f) for v,f in f_dict2.items()}
    return f_dict2, V_controller


def BN_control(B, f_dict, V_controller, T, initial_state:dict, control_sequence:dict):
    """初期状態に対して制御入力列を適用して時間遷移させる"""
    initial_state = {v:B.leaf1 if value else B.leaf0 for v,value in initial_state.items()}
    control_sequence = {v:B.leaf1 if value else B.leaf0 for v,value in control_sequence.items()}
    BNs = {0:initial_state}
    combination = {**initial_state, **control_sequence}
    
    f_dict2 = copy.deepcopy(f_dict)
    v_last = max(f_dict2)    # 現在の最後尾の変数名
    
    for t in range(1, T+1):
        # コントローラである変数に、新たな変数名を与える
        V_converter = {v:v_last+1+i for i,v in enumerate(V_controller)}
        f_dict_temp = dict()
        for v,f in f_dict2.items():
            f_temp = []
            for term in f:
                term_temp = [V_converter[abs(v)] * [-1,1][0<v] if abs(v) in V_controller else v for v in term]
                f_temp.append(term_temp)
            f_dict_temp[v] = f_temp
        
        # 時間を1進める
        BNs[t] = {v:B.calc_f(f,combination) for v,f in f_dict_temp.items()}
        combination = {**BNs[t], **control_sequence}
        v_last += len(V_controller)
    return BNs




def reduce_sheet(sheet):
    # 重複した行を削除
    sheet = np.unique(sheet, axis=0)
    # path_aの0,1の部分がpath_bで一致、もしくは-1のとき、path1を削除    !!! path_aの-1が一致することも必要
    del_idxs = []
    for idx_a,idx_b in itertools.permutations(range(len(sheet)), 2):
        if idx_a in del_idxs or idx_b in del_idxs:
            continue
        set_a0 = set(np.where(sheet[idx_a]==0)[0])
        set_b0_ = set(np.where((sheet[idx_b]==0) | (sheet[idx_b]==-1))[0])
        set_a1 = set(np.where(sheet[idx_a]==1)[0])
        set_b1_ = set(np.where((sheet[idx_b]==1) | (sheet[idx_b]==-1))[0])
        if set_a0 <= set_b0_ and set_a1 <= set_b1_:
            del_idxs.append(idx_a)
    return np.delete(sheet, del_idxs, 0)


def control1(B, node_u, V_controller, controller_info):
    """
    T時刻中で値を変更しない制御入力の個数が最大になる制御則を見つける
    """
    # 探索待ちのノードを変数名順で並べる
    nodes_next = defaultdict(lambda: defaultdict(lambda: np.empty((0,len(V_controller)), dtype='i1')))
    # 制御ノードの根を検出してsheetを持たせておく
    # sheet[row,V_controller.index(v)] = 0(値固定),1(値固定),-1(未定),-2(値変動)
    assert node_u.v in controller_info[:,0]
    nodes_next[node_u.v][node_u] = np.full((1,len(V_controller)), -1, dtype='i1')
    
    filled_sheets = []  # 葉1に到達したシート
    for v in controller_info[:,0]:
        # v: bdd上での制御ノードの変数名
        node_sheets = nodes_next.get(v)
        if node_sheets:
            # v_idx: vの元の変数名のV_controller上のインデックス
            v_idx = V_controller.index(controller_info[controller_info[:,0]==v, 1])
            for node,sheet in node_sheets.items():
                # sheetの簡単化
                sheet = reduce_sheet(sheet)
                # 
                # nodeの枝の選択をsheetに反映し子ノードに渡す
                for edge,node_c in enumerate(node[1:3]):
                    sheet_c = sheet.copy()
                    # -1 → edge
                    sheet_c[sheet[:,v_idx]==-1,v_idx] = edge
                    # ~edge → -2
                    sheet_c[sheet[:,v_idx]==1-edge,v_idx] = -2
                    
                    if isinstance(node_c,Leaf):
                        if node_c.value:
                            filled_sheets.append(sheet_c)
                    else:
                        # 子ノードをnodes_nextに登録
                        nodes_next[node_c.v][node_c] = np.concatenate([
                            nodes_next[node_c.v][node_c], sheet_c
                            ])
            
            del nodes_next[v]
    
    return reduce_sheet(np.concatenate(filled_sheets))








if __name__ == '__main__':
    # control1のテスト
    import BooleanNetwork, drawset, def_f
    B = BooleanNetwork.BooleanNetwork()
    
    """
    # 根が制御ノード
    T = 3
    V_controller = (1,2,3)
    v_last = 0
    n = len(V_controller) * T
    f_dict,_ = def_f.def_f('random','scalefree', n=n, gamma=0.001)
    f = [[2], [5, 7, 6], [1, -4, 9], [3]]
    """
    T = 2
    V_controller = (1,2,3)
    v_last = 2
    n = v_last + len(V_controller) * T
    f_dict,_ = def_f.def_f('random','scalefree', n=n, gamma=0.001)
    f = [[6, 1], [-5, -7, 8], [-3], [-4, 2]]
    #"""
    
    # bdd上の制御ノードの命名規則を整理
    controller_info = np.empty((len(V_controller)*T,3), dtype='i1')
    # bdd上の変数名
    controller_info[:,0] = np.arange(v_last+1, v_last+1+len(V_controller)*T)
    # 制御ノードとしての元の変数名
    controller_info[:,1] = np.tile(np.arange(1, 1+len(V_controller)), T)
    # 時刻
    controller_info[:,2] = [i//len(V_controller) for i in range(len(V_controller)*T)]
    
    
    node_u = B.calc_f(f,{})
    node_u = B.NOT(node_u)  # 積和系のでランダムに生成した式は1にたどり着きやすいので、反転する
    name_node = {
        int(v): f'u_{v_u}({t})'
        for v,v_u,t in controller_info
        }
    drawset.binary_tree(B.GetBDDdict(node_u),f'BDD_conv(t+{T})',name_node)
    
    #c1 = control1(B, node_u, V_controller, controller_info)
    c2 = control2(B, node_u, V_controller, controller_info)
    c3 = control3(B, node_u, V_controller, controller_info)
    path_num = B.CountPath(node_u)
    
    const = {
        v:c1[-1,V_controller.index(controller_info[controller_info[:,0]==v,1])] for v in range(1,1+n)
        if c1[-1,V_controller.index(controller_info[controller_info[:,0]==v,1])] in [0,1]
        }
    node_assigned = B.AssignConst(node_u, const)
    drawset.binary_tree(B.GetBDDdict(node_assigned),f'BDD_conv(t+{T})_assigned',name_node)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    