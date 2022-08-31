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
    parser.add_argument('--test_file', type=str, default='data/waic_nl2sql_testb_public.jsonl')
    parser.add_argument('--tag_file', type=str, default='data/tag.txt')
    parser.add_argument('--pretrain_model_path', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--save_model_path', type=str, default='model/ner')
    parser.add_argument('--max_seq_len', type=int, default=58)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ner_infere = NERInfere(args.tag_file, args.pretrain_model_path, args.save_model_path, args.max_seq_len, device)
    results = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            pred_entities = ner_infere.ner(data['question'])
            tmp = []
            for pred in pred_entities:
                if pred == "":
                    continue
                else:
                    tmp.append(pred)
            results.append(tmp)

    with open('data/ner_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
