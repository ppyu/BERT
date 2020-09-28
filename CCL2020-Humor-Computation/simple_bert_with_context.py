# -*- coding: utf-8 -*-
"""
@File   : simple_bert_with_context.py
@Author : Pengy
@Date   : 2020/8/12
@Description : Input your description here ... 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import sys
import json
import math
from random import sample

sys.path.append("../")
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf


class ModelConfig():
    """Configuration for Model"""

    def __init__(self,
                 data_dir,
                 vocab_file,
                 bert_config_file,
                 init_checkpoint,
                 output_dir,
                 train_batch_size=128,
                 learning_rate=2e-5,
                 num_train_epochs=3,
                 max_seq_length=128,
                 do_train=True,
                 do_eval=True
                 ):
        self.data_dir = data_dir
        self.vocab_file = vocab_file
        self.bert_config_file = bert_config_file
        self.init_checkpoint = init_checkpoint
        self.output_dir = output_dir
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.max_seq_length = max_seq_length
        self.do_train = do_train
        self.do_eval = do_eval

    @classmethod
    def from_dict(cls, json_object):
        config = ModelConfig()
        for key, value in enumerate(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.gfile.GFile(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))


class HumorDataProcessor():
    """
    幽默数据的处理
    """
    def __init__(self, context_size=2, split_rate=0.2):
        # 上下文窗口大小
        self.context_size = context_size
        self.split_rate = split_rate
        self.dev_inexs = []
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_exaples(self._read_tsv(os.path.join(data_dir, "cn_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_exaples(self._read_tsv(os.path.join(data_dir, "cn_train.csv")), "dev")

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1"]

    def _create_exaples(self, lines, set_type):
        num_dialogue = int(lines[-1][1])
        num_dev_dialogue = math.floor(num_dialogue * self.split_rate)
        # num_train_dialogue = num_dialogue - num_dev_dialogue
        if self.dev_indexs is not None:
            self.dev_indexs = sample(range(num_dialogue), num_dev_dialogue)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                # 第一行为列标题，忽略
                continue
            dialogue_id = line[1]
            if set_type == "dev" and int(dialogue_id) in self.dev_indexs:
                guid = "%s-%d" % (set_type, i)
                text_a = tokenization.convert_to_unicode(line[4])
                text_b = None
                text_b_len = FLAGS.max_seq_length - 3 - len(text_a)
                if text_b_len > 0:
                    text_b = tokenization.convert_to_unicode(
                        self._get_text_b(text_b_len, dialogue_id, i, lines))
                label = tokenization.convert_to_unicode(line[5])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            elif set_type == "train" and int(dialogue_id) not in self.dev_indexs:
                guid = "%s-%d" % (set_type, i)
                text_a = tokenization.convert_to_unicode(line[4])
                text_b = None
                text_b_len = FLAGS.max_seq_length - 3 - len(text_a)
                if text_b_len > 0:
                    text_b = tokenization.convert_to_unicode(
                        self._get_text_b(text_b_len, dialogue_id, i, lines))
                label = tokenization.convert_to_unicode(line[5])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _get_text_b(self, text_b_len, dialogue_id, index, lines):
        left = index - 1
        right = index + 1
        max_index = len(lines)
        text_b_list = []
        left_size = math.ceil(text_b_len / 2)
        right_size = text_b_len - left_size
        while True:
            if left == 0:
                break
            line = lines[left]
            if line[1] == dialogue_id:
                if len(line[4]) < left_size:
                    text_b_list.insert(0, line[4])
                    left_size -= len(line[4])
                    left -= 1
                else:
                    text_b_list.insert(0, line[4][-left_size:])
                    left_size -= left_size
                    left -= 1
                    break
            else:
                break
        if right_size > 0:
            while True:
                if right == max_index:
                    break
                line = lines[right]
                if line[1] == dialogue_id:
                    if len(line[4]) < right_size:
                        text_b_list.append(line[4])
                        right_size -= len(line[4])
                        right += 1
                    else:
                        text_b_list.append(line[4][:right_size])
                        right_size -= right_size
                        right += 1
                        break
                else:
                    break

        return ''.join(text_b_list)

class Model():
    def __init__(self, model_config_file, do_lower_case=True):
        self.config = ModelConfig.from_json_file(model_config_file)
        # cased表示区分大小写，uncased表示不区分大小写,do_lower_case表示是否进行小写处理
        self.do_lower_case = do_lower_case
        self.init_checkpoint = self.config["init_checkpoint"]
        self.bert_config = modeling.BertConfig.from_json_file(self.config["bert_config_file"])
        self.max_seq_length = self.config["max_seq_length"]
        if self.max_seq_length > self.bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self.max_seq_length, self.bert_config.max_position_embeddings))
        self.output_dir = self.config["output_dir"]
        self.vocab_file = self.config["vocab_file"]

    def build(self):
        # 创建模型输出路径，保存模型和中间结果
        tf.gfile.MkDir(self.output_dir)
        tokenization.validate_case_matches_checkpoint(self.do_lower_case, self.init_checkpoint)
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

    def train(self):
        pass

    def eval(self):
        pass

    def predicit(self):
        pass


if __name__ == '__main__':
    # 设置tensorflow的日志打印级别
    tf.logging.set_verbosity(tf.logging.INFO)
