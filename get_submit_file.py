import json
from string import ascii_letters


def judge_digit(s):
    if len(s) != 6:
        return False
    for each in s:
        if each not in '0123456789':
            return False
    return True


with open("data/dict.json", "r", encoding="utf-8") as f:
    mp = json.load(f)

with open("data/ner_result.json", "r", encoding="utf-8") as f:
    ner_result = json.load(f)

with open("data/classification_result.json", "r", encoding="utf-8") as f:
    classification_result = json.load(f)
    
assert len(ner_result) == len(classification_result)

s = set()
with open('data/sel2value.json', 'r', encoding='utf-8') as f:
    sel2value = json.load(f)
    for k, v in sel2value.items():
        for each in v:
            s.add(each)

with open('data/train_not_appear_mp.json', 'r', encoding='utf-8') as f:
    train_not_appear_mp = json.load(f)
    
for k, v in train_not_appear_mp.items():
    if v.lower() not in s and v.upper() not in s:
        print(v)
    s.add(v)

fix = 0
new_entity_in_submit = {}
with open("data/tmp.jsonl", "w", encoding="utf-8") as f, open('data/new_entity_in_submit.json', 'w', encoding='utf-8') as fw:
    for ner, each in zip(ner_result, classification_result):
        if len(ner) > 2:
            if '医疗' in ner and '生物' in ner:
                ner.remove('医疗')
                ner.remove('生物')
                ner.append('医疗;生物')
                print(each["question"])
                print(ner)
                print()
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
        each["sql"]["agg"] = [0]
        each["sql"]["limit"] = 0
        each["sql"]["orderby"] = []
        each["sql"]["asc_desc"] = 0
        
        if len(ner) == 0 or each["sql"]["cond_col"] == 100 or each["sql"]["cond_op"] == 100:  # TODO check
            values = []
        else:
            values = ner
            for i in range(len(values)):
                value = values[i]
                if '[SEP]' in value:  # 去掉[SEP]
                    values[i] = value.replace('[SEP]', '')
                if '\\n' in value:  # 去掉\n
                    values[i] = value.replace('\\n', '')
            
            if len(values) > 1 and ';'.join(sorted(values)) in train_not_appear_mp:
                new_entity_in_submit[';'.join(sorted(values))] = train_not_appear_mp[';'.join(sorted(values))]
                values = [train_not_appear_mp[';'.join(sorted(values))]]
                fix += 1
            else:
                for i in range(len(values)):
                    if values[i] in train_not_appear_mp:  # 对训练集中出现过的实体进行映射
                        new_entity_in_submit[values[i]] = train_not_appear_mp[values[i]]
                        fix += 1
                        values[i] = train_not_appear_mp[values[i]]
            
            each["sql"]["conds"] = []
            for value in values:
                each["sql"]["conds"].append([each["sql"]["cond_col"], each["sql"]["cond_op"], value])

        each["keywords"] = {
            "sel_cols": [mp["keywords_mp"][str(each["sql"]["sel"][0])]],
            "values": values
        }
        
        jsonstr = json.dumps(each, ensure_ascii=False)
        f.write(jsonstr + "\n")
    json.dump(new_entity_in_submit, fw, ensure_ascii=False, indent=4)
    
print('entity fix:', fix)


risk_fix = 0
# 处理风险相关
risk_d = {
    "中高风险": ['风险偏高', '中高风险'],
    "中低风险": ['中低风险'],
    "中风险": ['中等风险', '风险中', '中风险'],
    "高风险": ['风险比较高', '风险高', '高风险'],
    "低风险": ['稳定', '风险小', '稳健', '低风险', '比较稳', '安全系数高', '风险偏小', '风险低', '风险等级是稳', '稳赚不赔', '风险等级为稳', '稳建']
}

risk_keywords = []
for k, v in risk_d.items():
    risk_keywords += v

with open('data/tmp.jsonl', 'r', encoding='utf-8') as f, open('data/tmp2.jsonl', 'w', encoding='utf-8') as fw:
    for line in f:
        data = json.loads(line)
        question = data['question']
        appear_keywords = []
        for each in risk_keywords:
            if each in question:
                if each == '低风险':
                    idx = question.find('低风险')
                    if idx > 0 and question[idx-1] == '中':  # 避免中低风险 低风险重叠
                        continue
                if each == '风险中':
                    idx = question.find('风险中')
                    if idx > 0 and question[idx-1] == '低':  # 避免中低风险中
                        continue
                appear_keywords.append(each)
        
        if len(appear_keywords) == 3 and '稳定' in appear_keywords:
            appear_keywords.remove('稳定')
            
        if len(appear_keywords) == 2:
            a, b = appear_keywords[0], appear_keywords[1]
            if question.find(a) > question.find(b):
                a, b = b, a
            for k, v in risk_d.items():
                if a in v:
                    a2gt = k
                if b in v:
                    b2gt = k
            data['sql']['cond_conn_op'] = 2
            data['sql']['conds'] = [[7, 2, a2gt], [7, 2, b2gt]]
            data['keywords']['values'] = [a2gt, b2gt]
            risk_fix += 1
            
        if len(appear_keywords) == 1 and '估值' not in question:
            for value in data['keywords']['values']:
                if ('低' in value or '中' in value or '高' in value) and len(value) <= 4:
                    a = appear_keywords[0]
                    for k, v in risk_d.items():
                        if a in v:
                            a2gt = k
                    data['sql']['cond_conn_op'] = 0
                    data['sql']['conds'] = [[7, 2, a2gt]]
                    data['keywords']['values'] = [a2gt]
            risk_fix += 1
                        
        jsonstr = json.dumps(data, ensure_ascii=False)
        fw.write(jsonstr + "\n")

print('risk_fix:', risk_fix)


# 对齐cond_conn_op和conds
with open('data/tmp2.jsonl', 'r', encoding='utf-8') as f, open('data/tmp3.jsonl', 'w', encoding='utf-8') as fw:
    for line in f:
        data = json.loads(line)
        cond_conn_op = data['sql']['cond_conn_op']
        if 'conds' in data['sql']:
            conds = data['sql']['conds']
            values = data['keywords']['values']
            if cond_conn_op == 0:
                if len(conds) > 1 or len(values) > 1:
                    for value in values:
                        if len(value) == 0 or value in ['、', '1000', '中']:
                            data['keywords']['values'].remove(value)
                            for i in range(len(conds)):
                                if conds[i][2] == value:
                                    break
                            conds.pop(i)
                    if len(values) == 2:
                        find1 = find2 = False
                        for each in ascii_letters:
                            if each in values[0]:
                                find1 = True
                            if each in values[1]:
                                find2 = True
                        if find1 and find2:
                            data['sql']['cond_conn_op'] = 2
                    if len(values) > 1 and (not find1 or not find2):
                        for value in values:
                            find_tv = False
                            for tv in s:
                                if value in tv:
                                    find_tv = True
                                    break
                            if not find_tv:
                                data['keywords']['values'].remove(value)
                                for i in range(len(conds)):
                                    if conds[i][2] == value:
                                        break
                                conds.pop(i)
                        if len(data['keywords']['values']) == 0:
                            del data['sql']['conds']
                if data['sql']['cond_conn_op'] == 0 and (len(conds) > 1 or len(values) > 1):
                    data['sql']['cond_conn_op'] = 2
                    # print(data)
            elif cond_conn_op == 2:
                if len(conds) == 1 or len(values) == 1:
                    if '碳中和' in data['question']:
                        data['sql']['cond_conn_op'] = 0
                        for i, value in enumerate(values):
                            if value == '碳中和基金':
                                values[i] = '碳中和'
                    else:
                        if ('和' in conds[0][2] or '与' in conds[0][2]) and \
                            '碳中和' not in data['question'] and '民和股份' not in data['question'] and '润和软件' not in data['question'] and '诺安和鑫' not in data['question']:
                            if '和' in conds[0][2]:
                                a, b = conds[0][2].split('和')
                            elif '与' in conds[0][2]:
                                a, b = conds[0][2].split('与')
                            if len(a) and len(b):
                                if len(b) == 1 and b in ascii_letters and len(a) > 1 and a[-1] in ascii_letters:
                                    b = a[:-1] + b
                            if a in train_not_appear_mp:
                                a = train_not_appear_mp[a]
                            if b in train_not_appear_mp:
                                b = train_not_appear_mp[b]
                            if len(a) and len(b):
                                data['keywords']['values'] = [a, b]
                                data['sql']['conds'] = [[data['sql']['cond_col'], data['sql']['cond_op'], a], [data['sql']['cond_col'], data['sql']['cond_op'], b]]
                        else:
                            data['sql']['cond_conn_op'] = 0
                            
        jsonstr = json.dumps(data, ensure_ascii=False)
        fw.write(jsonstr + "\n")
