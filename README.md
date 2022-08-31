# 复现指南

1. `pip install -r requirements.txt` 安装依赖
2. `python read_data.py` 准备工作
3. `python get_sel2value.py` 准备工作
4. `python preprocess_ner.py` 准备ner训练数据
5. `python ner.py` 开始ner训练
6. `python get_train_not_appear_mp.py` 利用训练好的ner模型在训练集上做实体映射
7. `python ner_infere.py` ner预测
8. `python classification.py` 训练分类模型并预测 （训练和预测代码都在这个文件里，可以通过设置全局变量CFG中的do_train和do_test来决定是否 只做训练 或 只做测试 或 都做）
9. `python get_submit_file.py` 后处理得到的最终结果在 `data` 目录下的 `submit.jsonl` 文件



训练过程已全部设置好随机种子，完全可以复现出线上最佳结果。



# 总体思路

两部分：NER（实体识别）和文本分类，这两部分分别用一个模型解决

实体识别后需要进行实体映射，将部分实体进行对齐

文本分类采用五折交叉验证的方式训练

两部分的预训练语言模型均为 `chinese-roberta-wwm-ext` ，训练好的模型已存放在 `model` 目录下

