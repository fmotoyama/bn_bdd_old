# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:08:58 2022

@author: f.motoyama
"""
import re


f_dict = {}
label = [None]

with open('model.txt', encoding='utf-8') as f:
    s = f.read()
s = s.split('\n')

#左のスペース文字を区切りに、変数ラベル・式を抽出
for i,line in enumerate(s):
    temp = line.find(' ')
    label.append(line[:temp])   #式が示されているラベルのみを保持
    f_dict[i+1] = line[temp+1:]

#式を成型
pattern_not = re.compile(r'not \w+')

def replace_not(m):
    s = m.group()
    return '-' + s[4:]
        

for v,f in f_dict.items():
    #not ~をマイナス表記に置換
    temp = pattern_not.sub(replace_not, f)
    #演算子を置換
    temp = temp.replace(' and ', ',').replace(' or ', ' ').replace('(', '').replace(')', '')
    #文字列を数字の式に変換
    temp = temp.split()
    for i,term in enumerate(temp):
        term = term.split(',')
        term2 = []
        for l in term:
            sign = -1 if l[0] == '-' else 1
            l = l[1:] if l[0] == '-' else l
            if l in label:
                term2.append(label.index(l) * sign)
        temp[i] = term2
    temp = [term for term in temp if term]  #空の項を削除
    f_dict[v] = temp
    

label_sort = [None] + sorted(label[1:])

with open('f_dict.txt', mode='w') as f:
    f.write(str(f_dict))

temp = {i:l for i,l in enumerate(label) if l != None}
with open('label.txt', mode='w') as f:
    f.write(str(temp))
