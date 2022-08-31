import json
from collections import Counter
from transformers import BertTokenizer


MODEL = 'hfl/chinese-roberta-wwm-ext-large'
tokenizer = BertTokenizer.from_pretrained(MODEL)

questions = []
sels = []
aggs = []
limits = []
asc_descs = []
cond_conn_ops = []
conds_cols = []
conds_ops = []
conds_value = []
cnt = 0
max_len = max_q_len = 0
no_conds = 0
keywords_mp = {}

with open('data/waic_nl2sql_train.jsonl', 'r', encoding='utf-8') as f, open('data/train_not_appear_entity.txt', 'w', encoding='utf-8') as fw:
    for line in f:
        line = line.strip()
        data = json.loads(line)

        sel = data['sql']['sel'][0]
        sels.append(sel)

        agg = data['sql']['agg'][0]
        aggs.append(agg)

        limit = data['sql']['limit']
        limits.append(limit)

        if data['sql']['orderby'] != []:
            print(data['sql']['orderby'])

        asc_descs.append(data['sql']['asc_desc'])

        cond_conn_ops.append(data['sql']['cond_conn_op'])

        if 'conds' not in data['sql']:
            # print(data['question'])
            assert len(data['keywords']['values']) == 0
            no_conds += 1
            data['sql']['conds'] = [[100, 100, 100]]  # hacky

        if 'conds' in data['sql']:
            conds = data['sql']['conds']
            if len(conds) > 1:
                assert len(conds) == 2
                tmp, tmp2 = [], []
                for cond in conds:
                    tmp.append(cond[0])
                    tmp2.append(cond[1])
                for each in tmp:
                    assert each == tmp[0]
                for each in tmp2:
                    assert each == tmp2[0]
            for cond in conds:
                conds_cols.append(cond[0])
                conds_ops.append(cond[1])
                conds_value.append(cond[2])

        question = data['question']
        max_len = max(max_len, len(tokenizer(question)['input_ids']))
        max_q_len = max(max_q_len, len(question))
        values = data['keywords']['values']
        sel_col = data['keywords']['sel_cols'][0]
        keywords_mp[sel] = sel_col
        
        bug = False
        for value in values:
            if value not in question:
                bug = True
                break
        if bug:
            fw.write(question + '\t' + str(values) + '\n')
            cnt += 1

with open('data/waic_nl2sql_testa_public.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        data = json.loads(line)
        question = data['question']
        max_len = max(max_len, len(tokenizer(question)['input_ids']))


print('sel:', Counter(sels))
print('agg:', Counter(aggs))
print('limit:', Counter(limits))
print('asc_desc:', Counter(asc_descs))
print('cond_conn_op:', Counter(cond_conn_ops))
print('conds_col:', Counter(conds_cols))
print('conds_op:', Counter(conds_ops))
# print('conds_value:', Counter(conds_value))
print('cond value not appear in question num:', cnt)
print('max_len:', max_len)
print('max_q_len:', max_q_len)
print('no_conds:', no_conds)

# sel: Counter({1: 48200, 30: 3632, 29: 2856, 22: 2160, 46: 1496, 5: 1464, 3: 1448, 49: 760, 20: 752, 25: 744, 14: 744, 19: 736, 4: 736, 8: 736, 18: 736, 24: 736, 45: 720, 7: 720, 21: 720, 23: 712, 26: 712, 15: 712, 16: 696, 13: 688, 17: 688, 28: 688, 6: 680, 9: 680, 10: 672, 51: 672, 27: 672, 44: 152})
# agg: Counter({0: 78520})
# limit: Counter({0: 78520})
# asc_desc: Counter({0: 78520})
# cond_conn_op: Counter({0: 74040, 2: 4480})
# conds_col: Counter({30: 21619, 29: 17188, 1: 15332, 22: 12348, 46: 1361, 3: 1268, 5: 1255, 6: 742, 21: 697, 10: 691, 49: 690, 7: 686, 45: 665, 51: 659, 9: 653, 8: 634, 28: 617, 0: 364, 44: 134})
# conds_op: Counter({4: 69167, 2: 8436})
# cond value not appear in question: 3911

with open('data/dict.json', 'w', encoding='utf-8') as f:
    sels = Counter(sels)
    sels_mp = {}
    i = 0
    for k in sels.keys():
        sels_mp[k] = i
        i += 1
    sels_reverse_mp = {v: k for k, v in sels_mp.items()}
    
    cond_conn_ops = Counter(cond_conn_ops)
    cond_conn_ops_mp = {}
    i = 0
    for k in cond_conn_ops.keys():
        cond_conn_ops_mp[k] = i
        i += 1
    cond_conn_ops_reverse_mp = {v: k for k, v in cond_conn_ops_mp.items()}
    
    conds_cols = Counter(conds_cols)
    conds_cols_mp = {}
    i = 0
    for k in conds_cols.keys():
        conds_cols_mp[k] = i
        i += 1
    conds_cols_reverse_mp = {v: k for k, v in conds_cols_mp.items()}
    
    conds_ops = Counter(conds_ops)
    conds_ops_mp = {}
    i = 0
    for k in conds_ops.keys():
        conds_ops_mp[k] = i
        i += 1
    conds_ops_reverse_mp = {v: k for k, v in conds_ops_mp.items()}
    
    d = {
        'sels': sels_mp,
        'sels_reverse': sels_reverse_mp,
        'cond_conn_ops': cond_conn_ops_mp,
        'cond_conn_ops_reverse': cond_conn_ops_reverse_mp,
        'conds_cols': conds_cols_mp,
        'conds_cols_reverse': conds_cols_reverse_mp,
        'conds_ops': conds_ops_mp,
        'conds_ops_reverse': conds_ops_reverse_mp,
        'keywords_mp': keywords_mp
    }
    json.dump(d, f, ensure_ascii=False, indent=4)


