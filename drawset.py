# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:07:27 2020

@author: f.motoyama
"""

import os, itertools 
from copy import copy
from collections import defaultdict
import numpy as np
from graphviz import Digraph

from BDD import Leaf


def wiring_diagram(parent,fname='wiring_diagram'):
    """
    parent(dict型)のインタラクショングラフを描画する
    _typeにより以下のエラーの可能性
    CalledProcessError: Command '['dot.bat', '-Kneato', '-Tpng', '-O', 'test']' returned non-zero exit status 3221225477. [stderr: b'']
    """
    if 10 < len(parent):
        _type = 'sfdp'
    else:
        _type = 'neato'
        #_type = 'circo'
    #_type = 'sfdp'
    G = Digraph(format='png', engine=_type)
    G.attr('node', shape='circle')
    G.attr('graph', splines = 'curved', overlap = '0:')
    G.attr('edge', arrowsize = '0.5', color="#00000080")
    #"""
    
    for key in parent:
        for p in parent[key]:
            G.edge(str(p),str(key))
    #図を保存
    G.render(f'./figure/{fname}')
    os.remove(f'./figure/{fname}')


def wiring_diagram2(f_dict,fname='wiring_diagram'):
    """普通が→、notが●、両方が◆の矢印のインタラクショングラフを描画する"""
    #描画設定
    if 10 < len(f_dict):
        _type = 'sfdp'
    else:
        _type = 'neato'
    G = Digraph(format='png', engine=_type)
    G.attr('node', shape='circle')
    G.attr('graph', splines = 'curved', overlap = '0:')#scalexy
    G.attr('edge', arrowsize = '0.5', color="#00000080")
    
    #f_dictを読んで描画
    for xc in f_dict:
        #論理関数が定数のとき
        if type(f_dict[xc]) is int:
            G.node(str(xc))
            continue
        #正・負・混合で親ノードを分別
        parent = list(set(itertools.chain.from_iterable(f_dict[xc])))
        pos = set(); neg = set()
        for p in parent:
            if p < 0:
                neg.add(-p)
            else:
                pos.add(p)
        #混合は正・負の積集合 混合に該当したものを正・負から除く
        p_n = pos & neg
        pos = pos - p_n
        neg = neg - p_n
        
        #ノードの作成
        parent_class = [pos, neg, p_n]
        style = ['', 'dashed', 'dotted']
        #style = ['', '', '']
        for i in range(3):
            for xp in parent_class[i]:
                G.edge(str(xp), str(xc), style = style[i])
    
    #図を保存
    G.render(f'./figure/{fname}')
    os.remove(f'./figure/{fname}')






def binary_tree(bdd_dict, name='binary_tree', node_name={}):
    """BDDを描画する"""    
    G = Digraph(format='png')
    G.attr('node', shape='circle', fixedsize='true', width='0.75', fontsize='18')
    G.attr('edge', fontsize='20')
    
    if isinstance(bdd_dict,int):
        G.node(str(bdd_dict),str(bdd_dict),shape='square')
        G.render(f'./figure/{name}')
        os.remove(f'./figure/{name}')
        return
    # 葉ノード
    G.node('0','0',shape='square')
    G.node('1','1',shape='square')
    
    # 各変数の所属しているノード番地を変数ごとに格納
    V_group = defaultdict(list)
    for key,(v,_,_) in bdd_dict.items():
        v = node_name[v] if v in node_name else str(v)
        V_group[v].append(str(key))
    #変数ごとに行を整列
    for v,group in V_group.items():
        with G.subgraph() as s:
            s.attr(rank='same')
            for key in group:
                s.node(key,v)       #node(値：ノードの番地(bddでのkey), ラベル：変数名)
    
    # グラフを構成
    for key,(_,key0,key1) in bdd_dict.items():
        
        G.edge(str(key),str(key0),arrowhead='onormal',style='dashed')
        G.edge(str(key),str(key1),arrowhead='normal')
                
    #print(G)
    G.render(f'./figure/{name}')
    os.remove(f'./figure/{name}')




def transition_diagram(TT,fname='transition_diagram'): 
    """
    真理値表TT(2,2^n,n)を入力として、状態遷移図を描画する
    1111:0, 1110:1, 1101:2,...
    """
    temp = TT.shape
    l = temp[1]     #状態の個数
    n = temp[2]     #変数の個数
    
    if n <= 4:
        #変数が4つ以下のとき、2進数を表示
        ttl = np.empty(l,dtype = f'<U{n}')
        ttr = np.empty(l,dtype = f'<U{n}')
        for i in range(l):
            char_l = np.where(TT[0][i],'1','0')
            char_r = np.where(TT[1][i],'1','0')
            str_l = str()
            str_r = str()
            for j in range(n):
                str_l += char_l[j]
                str_r += char_r[j]
            ttl[i] = str_l
            ttr[i] = str_r
    else:
        #変数が5つ以上のとき、10進数に変換したものを表示
        digit = len(str(l-1))
        ttl = np.empty(l,dtype = f'<U{digit}')  #digit桁の文字列を格納
        ttr = np.empty(l,dtype = f'<U{digit}')
        TT2 = TT.astype(np.bool_)
        #Bool配列を文字列に変換
        for i in range(l):
            int_l = sum((2**i)*v for i,v in enumerate(np.flip(~TT2[0][i])))
            int_r = sum((2**i)*v for i,v in enumerate(np.flip(~TT2[1][i])))
            ttl[i] = str(int_l)
            ttr[i] = str(int_r)        
        
    
    G = Digraph(format='png',engine='twopi')#dot dot
    G.attr(rankdir='LR') #'TB'
    G.attr('node', shape='circle', fixedsize='true', width='0.75', fontsize='20')
    for i in range(l):
        if ttl[i] == ttr[i]:
            #G.attr('node',shape='doublecircle')    #二重丸にならないことがあるので下の文に改良
            G.node(ttl[i],shape='doublecircle',color='red')
            G.edge(ttl[i],ttr[i])
        else:
            #G.attr('node',shape='circle')
            G.edge(ttl[i],ttr[i])
           
    #print(G)
    G.render(f'./figure/{fname}')
    os.remove(f'./figure/{fname}')


def transition_diagram2(F,FF): 
    """PBNについて、真理値表の左側F、右側FF、遷移確率を入力として、状態遷移図を描画する"""
    l=len(F)        #状態の個数
    n=len(F[0])     #変数の個数
    f=[]; ff=[]     #F,FFのstr版
    for i in range(l):      #前状態
        temp=''
        for j in range(n):
            temp+=str(F[i][j])
        f+=[temp]
        ff+=[[]]
        for k in range(len(FF[i])):     #後状態
            ttemp=''
            for j in range(n):
                ttemp+=str(FF[i][k][j])
            ff[i]+=[[ttemp,str(FF[i][k][-1])]]

    G = Digraph(format='png',engine='dot')
    #G.attr(rankdir='TB') #'TB'
    G.attr('node', fontsize='20')
    for n in range(l):
        for nn in range(len(ff[n])):
            '''
            #固定点を強調
            if f[n]==ff[n][nn][0]:
                G.node(str(f[n]),shape='doublecircle',color='red')
                G.edge(str(f[n]),str(ff[n][nn][0]),label=ff[n][nn][1])
            else:
                G.attr('node',shape='circle')
                G.edge(str(f[n]),str(ff[n][nn][0]),label=ff[n][nn][1])
            '''
            G.attr('node',shape='circle')
            G.edge(str(f[n]),str(ff[n][nn][0]),label=ff[n][nn][1])
           
    #G.node('x1x2x3',shape='circle', fontsize='13', fontname='times-itaric')
    #print(G)
    G.render('./figure/transition_diagram')
    os.remove('./figure/transition_diagram')



def order_tree(h_order, vroot, fvs):
    """vrootに代入操作を行う計算順序を表す木構造を描画"""
    G = Digraph(format='png')
    G.attr('graph', rankdir='TB', constraint='false')#, ordering="out"
    G.attr('node', shape='circle', fixedsize='true', width='0.75', fontsize='20',style='filled',fillcolor='white')
    G.attr('edge', style = "dashed", penwidth='1')
    
    G.node('root',str(vroot),fillcolor='#efe4b0')
    
    #ノード名（index）の決め方　→　根：root,　fvs：count,　その他：変数名そのまま
    count = -1
    def scan(v,index):
        """vを、親ノードv2と接続する"""
        nonlocal count
        
        for vp in h_order[v][0]:
            if vp in fvs:
                G.node(str(count),str(vp),fillcolor='#efe4b0')
                G.edge(str(count),str(index))
                count -= 1
            else:
                G.edge(str(vp),str(index), style = "solid", penwidth='2')
                scan(vp,str(vp))
        
        for vp2 in h_order[v][1]:
            G.edge(str(vp2),str(index))
    
    scan(vroot,'root')
    G.render('./figure/order_tree')
    os.remove('./figure/order_tree')

def order_tree2(h_order, vroot, fvs):
    """数字なし"""
    G = Digraph(format='png')
    G.attr('graph', rankdir='LR', constraint='false')#, ordering="out"
    G.attr('node', shape='circle', fixedsize='true', width='0.75', fontsize='20',style='filled',fillcolor='white')
    G.attr('edge', style = "dashed", penwidth='1')
    
    G.node('root',str(),fillcolor='#efe4b0')
    
    #ノード名（index）の決め方　→　根：root,　fvs：count,　その他：変数名そのまま
    for v in h_order:
        if v not in fvs:
            G.node(str(v),str())
    count = -1
    def scan(v,index):
        """vを、親ノードv2と接続する"""
        nonlocal count
        
        for vp in h_order[v][0]:
            if vp in fvs:
                G.node(str(count),str(),fillcolor='#efe4b0')
                G.edge(str(count),str(index))
                count -= 1
            else:
                G.edge(str(vp),str(index), style = "solid", penwidth='2')
                scan(vp,str(vp))
        
        for vp2 in h_order[v][1]:
            G.edge(str(vp2),str(index))
    
    scan(vroot,'root')
    G.render('./figure/order_tree')
    os.remove('./figure/order_tree')


def order_tree3(h_order, vroot, fvs):
    """色付きorder_tree"""
    c = color()
    
    G = Digraph(format='png')
    G.attr('graph', rankdir='LR', constraint='false', ranksep='0.3')
    G.attr('node', shape='circle', fixedsize='true', width='0.75', fontsize='20',style='filled',fillcolor='white')
    G.attr('edge', style = "dashed", penwidth='1')
    
    G.node('root',str(vroot),fillcolor='#efe4b0')
    
    #ノード名（index）の決め方　→　根：root,　fvs：count,　その他：変数名そのまま
    count = -1
    pathcolor = c.rgbhex
    def scan(v,index):
        """vを、親ノードv2と接続する"""
        nonlocal count, pathcolor
        
        path = True    #分岐でないことを判定
        for vp in h_order[v][0]:
            if vp in fvs:
                G.node(str(count),str(vp),fillcolor='#eeeeee',shape='doublecircle')
                G.edge(str(count),str(index))
                count -= 1
            else:
                if path == False:
                    pathcolor = c.hue()
                path = False
                G.node(str(vp),fillcolor=pathcolor)
                G.edge(str(vp),str(index), style = "solid", penwidth='2')
                scan(vp,str(vp))
        
        for vp2 in h_order[v][1]:
            G.edge(str(vp2),str(index))
    
    scan(vroot,'root')    
    G.render('./figure/order_tree')
    os.remove('./figure/order_tree')



class color:
    def __init__(self):
        self.rgb = np.array([160,240,100])
        self.rgbhex = self.dec2hex(self.rgb)
        self.rgbmax = np.max(self.rgb)
        self.rgbmin = np.min(self.rgb)
        self.pos = 3 - np.argmax(self.rgb) - np.argmin(self.rgb)      #pos%3がargとなる
        self.sign = 1                       #符号
        self.step = 50
        
    def dec2hex(self, rgb):
        rgb = rgb.astype(np.int64)
        return '#' + hex(rgb[0])[2:].zfill(2) + hex(rgb[1])[2:].zfill(2) + hex(rgb[2])[2:].zfill(2)
    
    def hue(self):
        """彩度を1step進め、rgbを返す"""
        self.rgb[self.pos%3] += self.step * self.sign
        while self.rgb[self.pos%3] < self.rgbmin or self.rgbmax < self.rgb[self.pos%3]:
            #超過分（絶対値）
            op = min(abs(self.rgb[self.pos%3]-self.rgbmax),abs(self.rgb[self.pos%3]-self.rgbmin))
            #超過部分を修正
            self.sign *= -1
            self.rgb[self.pos%3] += op * self.sign
            #次の位置に分配
            self.pos += 1
            self.rgb[self.pos%3] += op * self.sign
        return self.dec2hex(self.rgb)
            
        
        


















