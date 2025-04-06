# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 15:06:51 2020

@author: f.motoyama
"""
from time import time
import numpy as np
from copy import copy
import itertools


def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

def QM(f):
    """
    fを最簡形にする
    f: 積和標準形を表す2次元list 変数名には0を用いてはいけない
    項の表現にset()を用いることで、行列で作業するより小さくなりやすい
    """
    if f in [0,1]:
        return f
    
    # fで用いられている変数を抽出
    V = set([abs(v) for v in itertools.chain.from_iterable(f)])
    l = len(V)          #変数の個数
    
    # fの各項をset型に変換
    ## 項内の同じ変数を削除
    f = [set(term) for term in f]
    ## 符号違いの変数をもつ項を削除
    f = [term for term in f if len(set(map(abs,term))) == len(term)]
    if f == []:
        return 0
    
    #fを主加法標準展開
    terms_minimum = []
    for term in f:
        v_add = V - {abs(v) for v in term}
        # v_addの変数の正負を変えるすべてのパターンを生成し、それぞれにtermを連結する
        term_list = [
            term | set(term_add)
            for term_add in itertools.product(*list((v,-v) for v in v_add))
            ]
        terms_minimum.extend(term_list)
    terms_minimum = get_unique_list(terms_minimum)
    
    #圧縮
    terms_principal = []      #主項
    table = terms_minimum
    for _ in range(1,l+1):
        #圧縮回数は最大l-1回だが、l=1の場合を処理するためl回の繰り返し
        table_next = []
        compressed = np.zeros(len(table), dtype = 'bool')   #圧縮できたらTrue
        for i,j in itertools.combinations(range(len(table)),2):
            # 符号の違う1つの変数以外が共通するとき、圧縮する
            symmetric_difference = list(table[i] ^ table[j])
            if len(symmetric_difference) == 2:
                if abs(symmetric_difference[0]) == abs(symmetric_difference[1]):
                    # tableに{v},{-v}が存在するとき、式は1で決定する
                    if len(table[i]) == 1:
                        return 1
                    compressed[[i,j]] = True
                    table_next.append(table[i] & table[j])
        
        #圧縮されなかった項を主項に追加
        for i in np.where(~compressed)[0]:
            terms_principal.append(table[i])
        
        if table_next == []:
            break
        table_next = get_unique_list(table_next)
        table = copy(table_next)
    
    
    #主項の変数と値が一致する最小項を探し、主項図に印をつける
    table_principal = np.zeros((len(terms_principal), len(terms_minimum)), dtype = 'bool')    #主項図
    for row,term_p in enumerate(terms_principal):
        for col,term_m in enumerate(terms_minimum):
            if term_m >= term_p:    #term_mがterm_pの部分集合のとき
                table_principal[row,col] = True
    
    
    #必須項を求める
    idx_m = np.where(np.count_nonzero(table_principal, axis=0) == 1)[0]     #単独の主項がカバーする最小項のidx
    idx_p = list(set(np.where(table_principal[:,idx_m])[0]))                #idx_mの最小項をカバーする主項のidx
    terms_essential = [terms_principal[i] for i in idx_p]
    #使われていない主項とカバーされていない主項図の領域を求める
    terms_principal2 = [terms_principal[i] for i in range(len(terms_principal)) if i not in idx_p]
    idx_m_noncover = np.where(np.sum(table_principal[idx_p], axis=0) != 1)[0]
    table_principal2 = table_principal[:,idx_m_noncover]        #カバーされていない最小項の列を抽出
    table_principal2 = np.delete(table_principal2, idx_p, 0)    #必須項の行を削除
    
    if len(table_principal2):
        #ぺトリック法 table_principal2の最小項をカバーするterms_principal2の主項を求める
        f_petric_ORAND = [np.where(column)[0].tolist() for column in table_principal2.T]  #ぺトリック方程式（和積系）
        #和積系を積和系に変換
        f_petric_ANDOR = [set()]
        for term_OR in f_petric_ORAND:
            f_petric_ANDOR2 = []
            for v in term_OR:
                f_petric_ANDOR2 += [term_AND | {v} for term_AND in f_petric_ANDOR]
            f_petric_ANDOR = get_unique_list(f_petric_ANDOR2)
        #積和系で最短の項を求める
        idx_principal2 = list()
        temp = len(terms_principal2)
        for term_AND in f_petric_ANDOR:
            if len(term_AND) < temp:
                idx_principal2 = term_AND
                temp = len(term_AND)
        
        terms_essential += [terms_principal2[i] for i in idx_principal2]
    
    terms_essential = [list(term) for term in terms_essential]
    return terms_essential



if __name__ == '__main__':
    
    f = [[-1,-2,3,4],[2,3,4],[1,2,-3],[1,-2,3,4]]
    #f = [[-1,2,-3,-4],[1,-2,-3,-4],[1,-2,3,-4],[1,-2,3,4],[1,2,-3,-4],[1,2,3,4]]
    #f = [[7]]
    f2 = QM(f)

