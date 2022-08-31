import os
import torch
import argparse
import json
from tqdm import tqdm

from ner_model import BiLSTMCRF
from ner_dataset import NERDataset
from ner_metrics import NERMetric


class NERInfere():
    def __init__(self, tag_file, pretrain_model_path, save_model_path, max_seq_len, device):
        self.device = device
        self.tag2idx, self.idx2tag = self.load_tagid(tag_file)
        self.model = BiLSTMCRF(hidden_dim=200, num_tags=len(self.tag2idx), model_path=pretrain_model_path, device=self.device)
        self.model.load_state_dict(torch.load(os.path.join(save_model_path, f"{pretrain_model_path.split('/')[-1]}.pt"), map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()
        self.dataset = NERDataset([], self.tag2idx, max_seq_len, pretrain_model_path)
        self.metrics = NERMetric(self.idx2tag)

    def load_tagid(self, tagfile):
        tag2idx, idx2tag = {}, {}
        with open(tagfile, 'r', encoding='utf-8') as fin:
            for line in fin:
                tag, idx = line.strip().split("\t")
                tag2idx[tag] = int(idx)
                idx2tag[int(idx)] = tag
        return tag2idx, idx2tag

    def ner(self, text):
        with torch.no_grad():
            tokens, token_ids, attention_mask = self.dataset.encode_text(text)
            token_ids = torch.tensor(token_ids).long().to(self.device)
            attention_mask = torch.tensor(attention_mask).long().to(self.device)
            logit = self.model(token_ids=token_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))[0]
            ent_info = self.metrics.get_entity(logit)
            entities = []
            for info in ent_info:
                etype, start, end = info
                entity = "".join(tokens[start:end + 1])
                entities.append(entity)
            return entities


def get_entity(text, labels):
        entities = []
        start = -1
        for index, tag in enumerate(labels):
            if tag.startswith("S"):
                entities.append(text[index])
            elif tag.startswith("B"):
                start = index
            elif tag.startswith("I"):
                pass
            else:
                if start != -1:
                    entities.append(text[start:index])
                start = -1
        return entities
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--not_appear_entity_file', type=str, default='data/train_not_appear_entity.txt')
    parser.add_argument('--tag_file', type=str, default='data/tag.txt')
    parser.add_argument('--pretrain_model_path', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--save_model_path', type=str, default='model/ner')
    parser.add_argument('--max_seq_len', type=int, default=58)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ner_infere = NERInfere(args.tag_file, args.pretrain_model_path, args.save_model_path, args.max_seq_len, device)

    mp = {}
    with open(args.not_appear_entity_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            q, gold_entities = line.split('\t')
            gold_entities = eval(gold_entities)
            pred_entities = ner_infere.ner(q)
            tmp = []
            for pred in pred_entities:
                if pred == "":
                    continue
                else:
                    tmp.append(pred)
            preds = []
            for each in tmp:
                if each in gold_entities:
                    gold_entities.remove(each)
                else:
                    preds.append(each)
            
            preds_str = ';'.join(sorted(preds))
            if len(preds_str) <= 1 or '风险' in preds_str or '低' in preds_str or '高' in preds_str or \
                '电子' in preds_str or '交通运输' in preds_str or '国债' in preds_str:
                continue
            
            preds = preds_str.split(';')
            if len(preds) == 1 and len(gold_entities) == 1:
                mp[preds[0]] = gold_entities[0]
            elif len(preds) == 2 and len(gold_entities) == 2:
                pred_a, pred_b = preds
                gold_a, gold_b = gold_entities
                if pred_a in gold_a:
                    mp[pred_a] = gold_a
                    mp[pred_b] = gold_b
                elif pred_a in gold_b:
                    mp[pred_a] = gold_b
                    mp[pred_b] = gold_a
                elif pred_b in gold_a:
                    mp[pred_a] = gold_b
                    mp[pred_b] = gold_a
                elif pred_b in gold_b:
                    mp[pred_a] = gold_a
                    mp[pred_b] = gold_b
            elif len(preds) == 2 and len(gold_entities) == 1:
                mp[preds_str] = gold_entities[0]
                
    with open('data/train_not_appear_mp.json', 'w', encoding='utf-8') as f:
        json.dump(mp, f, ensure_ascii=False, indent=4)
