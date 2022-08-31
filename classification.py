import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import *
from transformers import *
import json
import torch.nn.functional as F
import os
import random


CFG = { 
    'fold_num': 5,  # 交叉验证
    'model': 'hfl/chinese-roberta-wwm-ext',  # 预训练模型
    'max_len': 56,  # 文本截断的最大长度
    'epochs': 20,
    'train_bs': 32,  # batch_size，可根据自己的显存调整
    'valid_bs': 32,
    'lr': 5e-5,  # 学习率
    'num_workers': 8,
    'accum_iter': 1,  # 梯度累积
    'patience': 3,
    'do_train': True,
    'do_test': True,
    'seed': 42
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/dict.json', 'r', encoding='utf-8') as f:
    mp = json.load(f)

train_data = []
with open('data/waic_nl2sql_train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # {
        #     "id": 135042846, 
        #     "question": "涨幅最大的板块南方誉隆一年持有期混合a的净值", 
        #     "table_id": "FundTable", 
        #     "sql": {
        #         "sel": [13], 
        #         "agg": [0], 
        #         "limit": 0, 
        #         "orderby": [], 
        #         "asc_desc": 0, 
        #         "cond_conn_op": 0, 
        #         "conds": [[1, 4, "南方誉隆一年持有期混合a"]]
        #     }, 
        #     "keywords": {
        #         "sel_cols": ["净值"], 
        #         "values": ["南方誉隆一年持有期混合a"]
        #     }
        # }
        data = json.loads(line)
        
        if 'conds' not in data['sql']:
            data['sql']['conds'] = [[100, 100, 100]]  # hacky
        
        data['sel'] = mp['sels'][str(data['sql']['sel'][0])]
        data['cond_conn_op'] = mp['cond_conn_ops'][str(data['sql']['cond_conn_op'])]
        data['cond_col'] = mp['conds_cols'][str(data['sql']['conds'][0][0])]
        data['cond_op'] = mp['conds_ops'][str(data['sql']['conds'][0][1])]
        del data['sql']
        del data['table_id']
        del data['keywords']
        train_data.append(data)
        
test_data = []
with open('data/waic_nl2sql_testb_public.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        data = json.loads(line)
        data['sel'] = -1
        data['cond_conn_op'] = -1
        data['cond_col'] = -1
        data['cond_op'] = -1
        test_data.append(data)

tokenizer = BertTokenizer.from_pretrained(CFG['model'])


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        sel = self.data[idx]['sel']
        cond_conn_op = self.data[idx]['cond_conn_op']
        cond_col = self.data[idx]['cond_col']
        cond_op = self.data[idx]['cond_op']
        return question, sel, cond_conn_op, cond_col, cond_op


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.model = AutoModel.from_pretrained(CFG['model'])
        self.config = self.model.config
        self.sel_head = nn.Linear(self.config.hidden_size, len(mp['sels']))
        self.cond_conn_op_head = nn.Linear(self.config.hidden_size, len(mp['cond_conn_ops']))
        self.cond_col_head = nn.Linear(self.config.hidden_size, len(mp['conds_cols']))
        self.cond_op_head = nn.Linear(self.config.hidden_size, len(mp['conds_ops']))
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids, attention_mask, token_type_ids)
        cls = outputs.last_hidden_state[:, 0, :]
        sel_logits = self.sel_head(cls)
        cond_conn_op_logits = self.cond_conn_op_head(cls)
        cond_col_logits = self.cond_col_head(cls)
        cond_op_logits = self.cond_op_head(cls)
        return sel_logits, cond_conn_op_logits, cond_col_logits, cond_op_logits


def collate_fn(data): 
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[0], padding='max_length', truncation=True, max_length=CFG['max_len'], return_tensors='pt')
        input_ids.append(text['input_ids'][0])
        attention_mask.append(text['attention_mask'][0])
        token_type_ids.append(text['token_type_ids'][0])
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    token_type_ids = torch.stack(token_type_ids)
    sel_labels = torch.tensor([x[1] for x in data], dtype=torch.long)
    cond_conn_op_labels = torch.tensor([x[2] for x in data], dtype=torch.long)
    cond_col_labels = torch.tensor([x[3] for x in data], dtype=torch.long)
    cond_op_labels = torch.tensor([x[4] for x in data], dtype=torch.long)
    return input_ids, attention_mask, token_type_ids, \
        sel_labels, cond_conn_op_labels, cond_col_labels, cond_op_labels


def train_model(model, train_loader):  # 训练一个epoch
    model.train()
    optimizer.zero_grad()
    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for step, (input_ids, attention_mask, token_type_ids, sel_labels, cond_conn_op_labels, cond_col_labels, cond_op_labels) in enumerate(tk):
        input_ids, attention_mask, token_type_ids, sel_labels, cond_conn_op_labels, cond_col_labels, cond_op_labels = \
            input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), sel_labels.to(device), cond_conn_op_labels.to(device), cond_col_labels.to(device), cond_op_labels.to(device)
        
        sel_logits, cond_conn_op_logits, cond_col_logits, cond_op_logits = model(input_ids, attention_mask, token_type_ids)
        sel_loss = criterion(sel_logits, sel_labels) / CFG['accum_iter']
        cond_conn_op_loss = criterion(cond_conn_op_logits, cond_conn_op_labels) / CFG['accum_iter']
        cond_col_loss = criterion(cond_col_logits, cond_col_labels) / CFG['accum_iter']
        cond_op_loss = criterion(cond_op_logits, cond_op_labels) / CFG['accum_iter']
        loss = sel_loss + cond_conn_op_loss + cond_col_loss + cond_op_loss
        loss.backward()

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


def test_model(model, val_loader):
    model.eval()
    sel_acc = cond_conn_op_acc = cond_col_acc = cond_op_acc = total = 0
    
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for step, (input_ids, attention_mask, token_type_ids, sel_labels, cond_conn_op_labels, cond_col_labels, cond_op_labels) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, sel_labels, cond_conn_op_labels, cond_col_labels, cond_op_labels = \
            input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), sel_labels.to(device), cond_conn_op_labels.to(device), cond_col_labels.to(device), cond_op_labels.to(device)
        
            sel_logits, cond_conn_op_logits, cond_col_logits, cond_op_logits = model(input_ids, attention_mask, token_type_ids)

            sel_acc += (sel_logits.argmax(1) == sel_labels).sum().item()
            cond_conn_op_acc += (cond_conn_op_logits.argmax(1) == cond_conn_op_labels).sum().item()
            cond_col_acc += (cond_col_logits.argmax(1) == cond_col_labels).sum().item()
            cond_op_acc += (cond_op_logits.argmax(1) == cond_op_labels).sum().item()
            total += sel_labels.size(0)
            
    return sel_acc / total, cond_conn_op_acc / total, cond_col_acc / total, cond_op_acc / total


if CFG['do_train']:
    criterion = nn.CrossEntropyLoss()

    sels = []
    for each in train_data:
        sels.append(each['sel'])
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True).split(np.arange(len(train_data)), sels)  # TODO

    cv = []  # 保存每折的最佳准确率

    for fold, (trn_idx, val_idx) in enumerate(folds):
        print('fold:', fold)
        
        train = []
        val = []
        for i in trn_idx:
            train.append(train_data[i])
        for i in val_idx:
            val.append(train_data[i])

        train_set = MyDataset(train)
        val_set = MyDataset(val)

        train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True, num_workers=CFG['num_workers'])
        val_loader = DataLoader(val_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False, num_workers=CFG['num_workers'])

        best_acc = 0
        no_update = 0

        model = ClassificationModel().to(device)

        optimizer = AdamW(model.parameters(), lr=CFG['lr'])
        # get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'], CFG['epochs'] * len(train_loader) // CFG['accum_iter']) 

        for epoch in range(CFG['epochs']):
            print('epoch:', epoch)

            train_model(model, train_loader)
            sel_acc, cond_conn_op_acc, cond_col_acc, cond_op_acc = test_model(model, val_loader)
            print('sel_acc, cond_conn_op_acc, cond_col_acc, cond_op_acc:', sel_acc, cond_conn_op_acc, cond_col_acc, cond_op_acc)
            all_acc = sel_acc + cond_conn_op_acc + cond_col_acc + cond_op_acc
            
            if all_acc > best_acc:
                print('new best acc:', all_acc)
                no_update = 0
                best_acc = all_acc
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), 'model/{}_fold_{}.pt'.format(CFG['model'].split('/')[-1], fold))
                else:
                    torch.save(model.state_dict(), 'model/{}_fold_{}.pt'.format(CFG['model'].split('/')[-1], fold))
            else:
                no_update += 1
                print('no update:', no_update)
                if no_update == CFG['patience']:
                    print('patience reached, stop training')
                    break
                
        cv.append(best_acc)

    print(np.mean(cv))
    print(cv)
# 3.979865002547122
# [3.98070555272542, 3.9754839531329598, 3.98191543555782, 3.9803234844625575, 3.980896586856852]

if CFG['do_test']:
    test_set = MyDataset(test_data)
    test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False, num_workers=CFG['num_workers'])

    model = ClassificationModel().to(device)
    model.eval()

    sel_predictions = []
    cond_conn_op_predictions = []
    cond_col_predictions = []
    cond_op_predictions = []
    for fold in range(CFG['fold_num']):
        sel_probs = []
        cond_conn_op_probs = []
        cond_col_probs = []
        cond_op_probs = []
        model.load_state_dict(torch.load('model/{}_fold_{}.pt'.format(CFG['model'].split('/')[-1], fold)))
        with torch.no_grad():
            tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
            for step, (input_ids, attention_mask, token_type_ids, sel_labels, cond_conn_op_labels, cond_col_labels, cond_op_labels) in enumerate(tk):
                input_ids, attention_mask, token_type_ids, sel_labels, cond_conn_op_labels, cond_col_labels, cond_op_labels = \
                input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), sel_labels.to(device), cond_conn_op_labels.to(device), cond_col_labels.to(device), cond_op_labels.to(device)
                sel_logits, cond_conn_op_logits, cond_col_logits, cond_op_logits = model(input_ids, attention_mask, token_type_ids)

                sel_probs.extend(F.softmax(sel_logits, dim=-1).cpu().numpy().tolist())
                cond_conn_op_probs.extend(F.softmax(cond_conn_op_logits, dim=-1).cpu().numpy().tolist())
                cond_col_probs.extend(F.softmax(cond_col_logits, dim=-1).cpu().numpy().tolist())
                cond_op_probs.extend(F.softmax(cond_op_logits, dim=-1).cpu().numpy().tolist())
        
        sel_predictions.append(sel_probs)
        cond_conn_op_predictions.append(cond_conn_op_probs)
        cond_col_predictions.append(cond_col_probs)
        cond_op_predictions.append(cond_op_probs)

    sel_predictions = np.mean(sel_predictions, 0).argmax(1)
    cond_conn_op_predictions = np.mean(cond_conn_op_predictions, 0).argmax(1)
    cond_col_predictions = np.mean(cond_col_predictions, 0).argmax(1)
    cond_op_predictions = np.mean(cond_op_predictions, 0).argmax(1)

    result = []
    for i, each in enumerate(test_data):
        # {
        #     "id": 135042846, 
        #     "question": "涨幅最大的板块南方誉隆一年持有期混合a的净值", 
        #     "table_id": "FundTable", 
        #     "sql": {
        #         "sel": [13], 
        #         "agg": [0], 
        #         "limit": 0, 
        #         "orderby": [], 
        #         "asc_desc": 0, 
        #         "cond_conn_op": 0, 
        #         "conds": [[1, 4, "南方誉隆一年持有期混合a"]]
        #     }, 
        #     "keywords": {
        #         "sel_cols": ["净值"], 
        #         "values": ["南方誉隆一年持有期混合a"]
        #     }
        # }
        each['sql'] = {}
        each['sql']['sel'] = [mp['sels_reverse'][str(sel_predictions[i])]]
        each['sql']['cond_conn_op'] = mp['cond_conn_ops_reverse'][str(cond_conn_op_predictions[i])]
        each['sql']['cond_col'] = mp['conds_cols_reverse'][str(cond_col_predictions[i])]
        each['sql']['cond_op'] = mp['conds_ops_reverse'][str(cond_op_predictions[i])]
        del each['sel']
        del each['cond_conn_op']
        del each['cond_col']
        del each['cond_op']
        result.append(each)
    
    with open('data/classification_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    