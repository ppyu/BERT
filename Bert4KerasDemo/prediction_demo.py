# -*- coding: utf-8 -*-
"""
@File   : prediction_demo.py
@Author : Pengy
@Date   : 2020/9/23
@Description : Input your description here ... 
"""
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
import keras

config_path = "../chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "../chinese_L-12_H-768_A-12/bert_model.ckpt"
dict_path = "../chinese_L-12_H-768_A-12/vocab.txt"

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 建立模型，加载权重
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)
# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
print(token_ids, segment_ids)

print('\n{} predicting {}'.format('=' * 5, '=' * 5))
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))

model.summary()

print('\n===== reloading and predicting =====\n')
model.save('test.model')
del model
model = keras.models.load_model('test.model')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
