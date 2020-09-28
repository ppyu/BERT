# -*- coding: utf-8 -*-
"""
@File   : get_test_result_into_csv.py
@Author : Pengy
@Date   : 2020/8/13
@Description : Input your description here ... 
"""

import pandas as pd
import tensorflow as tf
import csv

cn_result_tsv = "./cn_test_results.tsv"
en_result_tsv = "./en_test_results.tsv"

cn_output_csv = "./cn_NLPers.csv"
en_output_csv = "./en_NLPers.csv"

with tf.gfile.Open(cn_result_tsv, "r") as f:
    # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    # 以逗号作为分隔符
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    out_lines = [["ID", "Label"]]
    id = 0
    for line in reader:
        label_0 = line[0]
        label_1 = line[1]
        if float(label_0) > float(label_1):
            label = 0
        else:
            label = 1
        out_lines.append([id, label])
        id += 1
    out_csv = tf.gfile.Open(cn_output_csv, "w")
    csv_writer = csv.writer(out_csv)
    csv_writer.writerows(out_lines)
    out_csv.close()

with tf.gfile.Open(en_result_tsv, "r") as f:
    # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    # 以逗号作为分隔符
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    out_lines = [["ID", "Label"]]
    id = 0
    for line in reader:
        label_0 = line[0]
        label_1 = line[1]
        if float(label_0) > float(label_1):
            label = 0
        else:
            label = 1
        out_lines.append([id, label])
        id += 1
    out_csv = tf.gfile.Open(en_output_csv, "w")
    csv_writer = csv.writer(out_csv)
    csv_writer.writerows(out_lines)
    out_csv.close()
