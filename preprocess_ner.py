import os
import json
import argparse
import random


random.seed(42)


def load_data(inputfile, output_train_file, output_dev_file):
    pairs = []
    with open(inputfile, 'r', encoding='utf-8') as fin:
        for line in fin:
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
            question = data['question']
            values = data['keywords']['values']
            sequence_label(question, values, pairs)

    random.shuffle(pairs)
    
    num = len(pairs)
    
    with open(output_train_file, 'w', encoding='utf-8') as fout:
        json.dump(pairs[:int(0.8*num)], fout, ensure_ascii=False, indent=4)
    
    with open(output_dev_file, 'w', encoding='utf-8') as fout:
        json.dump(pairs[int(0.8*num):], fout, ensure_ascii=False, indent=4)


def sequence_label(sequence, entities, pairs):
    tags = ["O"] * len(sequence)
    for entity in entities:
        if entity in sequence:
            start_index = get_sequence_labels(sequence, entity, tags)
        else:
            return
    sample = {"text": sequence, "labels": tags}
    pairs.append(sample)


def get_sequence_labels(sequence, entity, tags):
    index = sequence.find(entity)
    start_index = []
    if index != -1:
        if tags[index] == "O":
            if len(entity) == 1:
                tags[index] = "S"
            else:
                tags[index] = "B"
            start_index.append(index)
        for i in range(index + 1, index + len(entity)):
            if tags[i] == "O":
                tags[i] = "I"
    return start_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="data")
    parser.add_argument('--output_path', type=str, default="data")
    args = parser.parse_args()

    train_file = os.path.join(args.input_path, "waic_nl2sql_train.jsonl")
    output_train_file = os.path.join(args.output_path, "ner_train.json")
    output_dev_file = os.path.join(args.output_path, "ner_dev.json")
    load_data(train_file, output_train_file, output_dev_file)


if __name__ == "__main__":
    main()
    