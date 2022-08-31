import json
import pandas as pd
import re
from collections import defaultdict


df = pd.read_excel('data/fundTable.xlsx')
print(len(df.columns))

df = df.drop(columns=['产品ID:key', '基金经理从业年限'])
print(len(df.columns))

with open('data/dict.json', 'r', encoding='utf-8') as fin:
    mp = json.load(fin)
mp = mp['keywords_mp']

d = {}
for k, v in mp.items():
    k = int(k)
    c = df.iloc[:, k]
    d[v] = list(set(c.values.tolist()))
del d['净值']
del d['三个月夏普率']
del d['近一周涨跌幅']
del d['近六个月涨跌幅']
del d['一个月夏普率']
del d['六个月夏普率']
del d['基金规模']
del d['近一个月涨跌幅']
del d['近三个月涨跌幅']
del d['一年夏普率']
del d['昨日涨跌幅']
del d['成立以来涨跌幅']
del d['成立以来夏普率']
del d['近一年涨跌幅']

s = set()
new_d = defaultdict(set)
for k, v in d.items():
    for each in v:
        try:
            if each[0] == '[':
                each = each[1:-1]
                each = re.split('，|,', each)
                for each_ in each:
                    each_ = each_.strip()
                    s.add(each_)
                    new_d[k].add(each_)
            else:
                each = each.strip()
                s.add(each)
                new_d[k].add(each)
        except:
            print(each)

for k, v in new_d.items():
    new_d[k] = list(v)

with open('data/sel2value.json', 'w', encoding='utf-8') as fout:
    json.dump(new_d, fout, ensure_ascii=False, indent=4)
