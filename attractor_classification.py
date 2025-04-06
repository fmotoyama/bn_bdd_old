# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:25:50 2024

@author: fmotoyama
"""
import pickle, datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from BooleanNetwork import BooleanNetwork
import drawset, def_f

rng = np.random.default_rng(0)


def state2idx(state: dict):
    return sum((2**i)*v for i,v in enumerate(np.flip(~state)))
def idx2state(idx, l):
    return ~np.array([int(v) for v in format(idx,f'0{l}b')], dtype=np.bool_)

def GetAttractor(B, BN):
    """総当たりでアトラクターを求める"""
    V = BN.keys()
    l = len(V)
    
    # 真理値表の右側を得る
    TT = B.BN_to_TT(BN).astype(np.bool_)
    TTr_idx = np.array([state2idx(state) for state in TT[1]])
    
    sheet = np.zeros(2**l, dtype=np.int16)   # 各状態がどのアトラクターに到達するかを記す
    attractors = []
    attractor_id = 1
    tgt = 0     # 探索対象の状態
    path = []
    while True:
        if sheet[tgt]:
            # 探索済みの状態に到達したとき、pathをそこにつなげる
            sheet[path] = sheet[tgt]
        elif tgt in path:
            # ループ検知のとき、アトラクターを記録し、pathを探索済みとする
            attractors.append(path[path.index(tgt):])
            sheet[path] = attractor_id
            attractor_id += 1
        else:
            # それ以外のとき、状態遷移して続行
            path.append(tgt)
            tgt = TTr_idx[tgt]
            continue
        
        # 新しいスタート状態の生成
        tgts = np.where(sheet == 0)[0]
        if tgts.size == 0:
            break
        tgt = tgts[0]
        path = []
    
    sheet -= 1  # attractor_idを0始まりにする
    Y = np.identity(len(attractors), dtype=np.int8)[sheet]
    Dataset = {'X':TT[0], 'Y':Y}
    return attractors, Dataset


def MakeDataset(B, f_dict, N):
    """状態とアトラクターのデータセットを作る"""
    V = list(f_dict)
    l = len(V)
    
    # 初期状態セット
    assert N <= 2**l
    X_idx = set(); temp = 2**l
    while len(X_idx) < N:
        X_idx.add(rng.integers(temp)) 
    X = np.array([idx2state(x_idx,l) for x_idx in X_idx])
    
    def state2bdd(state: np.ndarray):
        bdd = B.leaf1
        for var,value in enumerate(state):
            bdd &= B.GetUnitNode(var+1) if value else ~B.GetUnitNode(var+1)
        return bdd
    X_bdd = [state2bdd(state) for state in X]
    
    # 各アトラクターのBasinを得る
    B.SetTransition(BN)
    attractors_bdd = B.GetAttractor(f_dict)
    attractors = [B.EnumPath(attractor_bdd, V)[0] for attractor_bdd in attractors_bdd]
    basins_bdd = [B.GetBasin(BN, attractor_bdd) for attractor_bdd in attractors_bdd]
    print('basin size:',[B.CountPath(basin_bdd,V) for basin_bdd in basins_bdd])
    # xがどのアトラクターのBasinに属するか求める
    Y = []
    for x_bdd in X_bdd:
        for id_,basin_bdd in enumerate(basins_bdd):
            if x_bdd & basin_bdd != B.leaf0:
                Y.append(id_)
                break
        else:
            raise Exception
    Y = np.identity(len(attractors), dtype=np.int8)[Y]
    Dataset = {'X':X, 'Y':Y}
    return attractors, Dataset
    
    


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, config.output_size) #pytorchの仕様のため、出力層の活性化関数は省略
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x    


if __name__ == '__main__':
    B = BooleanNetwork()
    
    name = 'f_dict_30'
    #f_dict,_ = def_f.def_f('random','normal',n=18)
    #f_dict,_ = def_f.def_f('import','f_dict_5')
    #f_dict,_ = def_f.def_f('import','f_dict_15')
    #f_dict,_ = def_f.def_f('import','f_dict_18')
    f_dict,_ = def_f.def_f('import',name)
    B.save_f_dict(f_dict)
    BN = B.GetBN(f_dict)
    
    class Config:
        def __init__(self, name):
            self.save_path = Path("checkpoints") / (f'{name}_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            self.load_file = None
            #self.load_file = Path("checkpoints") / 'f_dict_15_2024-09-06-15-27-59/Dataset.pkl'
            # データサイズ
            self.N = 1_000_000
            # Network
            self.input_size = len(f_dict)
            self.hidden_size = 64
            self.output_size = None
            # Learner
            self.use_gpu = True
            self.steps = 100_000
            self.batch_size = 32
            # logger
            self.num_logging = 500  # ログをとる回数
    config = Config(name)
    # 記録用のディレクトリを作成
    config.save_path.mkdir(parents=True)
    # 設定の保存
    with open(config.save_path/'config.pkl', mode="wb") as f:
        pickle.dump(config, f)
    # アトラクター情報の呼び出し
    if config.load_file:
        with open(config.load_file, 'rb') as f:
            #attractors,Dataset = pickle.load(f)
            a = pickle.load(f)
    
    # アトラクターを求める
    #attractors, Dataset = GetAttractor(B, BN)
    attractors, Dataset = MakeDataset(B, f_dict, config.N)
    #drawset.wiring_diagram(B.f_dict_to_parent(f_dict), 'wd_n=5')
    #drawset.transition_diagram(B.BN_to_TT(BN),'td_n=5')
    #アトラクター情報を保存
    with open(config.save_path/'Dataset.pkl', 'wb') as f:
        pickle.dump((attractors,Dataset), f)
    
    # アトラクター情報で学習準備
    config.output_size = len(attractors)
    # データセット用意
    device = 'cuda' if config.use_gpu else 'cpu'
    train_X, test_X, train_Y, test_Y = train_test_split(Dataset['X'], Dataset['Y'], test_size=0.2) #データを8:2に分割
    train_X = torch.tensor(train_X, device=device).float()
    train_Y = torch.tensor(train_Y, device=device).float()
    test_X = torch.tensor(test_X, device=device).float()
    #test_Y = torch.tensor(test_Y, device=device).float()
    
    train_dataloader = DataLoader(
        TensorDataset(train_X, train_Y), 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    # 学習
    net = Net(config).to(device=device).float()
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    
    losses = []
    #訓練ループ
    step = 0
    while step < config.steps:
        for batch, label in train_dataloader:
            optimizer.zero_grad()
            t_p = net(batch)
            loss = criterion(t_p,label)
            loss.backward()
            optimizer.step()
            
            step += 1
            if step % (config.steps // config.num_logging) == 0:
                losses.append(loss.to('cpu').detach().numpy())   #y軸方向のリストに損失の値を代入
            if step % (config.steps // 10) == 0:  # 10回、損失の値を表示
                print("step: %d  loss: %f" % (step ,float(loss)))
            if config.steps <= step:
                break
            
    
    
    #損失の描画
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.plot(
        list(range(0, config.steps, config.steps//config.num_logging)),
        losses
        )
    #plt.show()
    plt.savefig(config.save_path/'loss.png')
    
    # 評価
    with torch.no_grad():   # 試験用データでは勾配を計算しない
        eval_Y = net(test_X).to('cpu').detach().numpy()
    eval_Y = np.argmax(eval_Y, axis=1)
    
    check = np.array([np.argmax(test_y) == eval_y for test_y,eval_y in zip(test_Y, eval_Y)], dtype=np.bool_)
    print(f'正解率 {100 * np.sum(check) / len(test_Y)} %')
    print(f'不正解 {np.sum(~check)} / {len(test_Y)}')
    
    

    
