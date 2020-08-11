import tensorflow as tf
from tensorflow import keras
import jieba
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense
import sklearn
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

train_dev_data_path = "./train_dev_data/"


class Validation_Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            validate_data_x))).round()
        val_targ = validate_data_y
        _val_f1 = f1_score(val_targ, val_predict, pos_label=1)
        _val_recall = recall_score(val_targ, val_predict, pos_label=1)
        _val_precision = precision_score(val_targ, val_predict, pos_label=1)
        _val_acc = accuracy_score(val_targ, val_predict)

        print('\nf1:', _val_f1, ', recall:', _val_recall, ', precision:', _val_precision, ', accuracy:', _val_acc)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return
        # val_predict = (np.asarray(self.model.predict(
        #     self.validation_data[0]))).round()
        # val_targ = self.validation_data[1]
        # _val_f1 = f1_score(val_targ, val_predict)
        # _val_recall = recall_score(val_targ, val_predict)
        # _val_precision = precision_score(val_targ, val_predict)
        # print('验证集--- f1:', _val_f1, ', recall:', _val_recall, ', precision:', _val_precision)
        # self.val_f1s.append(_val_f1)
        # self.val_recalls.append(_val_recall)
        # self.val_precisions.append(_val_precision)
        # return


def get_data(path):
    data = pd.read_csv(path)
    return data


# 利用jieba对句子进行分词
def cut_sentence(sentence):
    # 使用jieba的默认精确模式进行分词
    words = list(jieba.cut(sentence))
    return words


def prapare_with_tokenize(path):
    data_df = pd.read_csv(path)
    sentence_len_list = []
    word_set = set()
    for sentence in data_df["Sentence"]:
        word_set = word_set.union(set(cut_sentence(sentence)))
        sentence_len_list.append(len(cut_sentence(sentence)))

    print("总词汇量：", len(word_set))

    print("分词后最小长度", min(sentence_len_list))
    print("分词后中间长度", sorted(sentence_len_list)[len(sentence_len_list) // 2])
    print("分词后最大长度", max(sentence_len_list))
    vcab_size = len(word_set)
    max_length = max(sentence_len_list)
    return data_df, word_set, vcab_size, max_length


def prapare_without_tokenize(path):
    data_df = pd.read_csv(path)
    sentence_len_list = []
    word_set = set()
    for sentence in data_df["Sentence"]:
        word_set = word_set.union(set(sentence))
        sentence_len_list.append(len(sentence))

    print("总词汇量：", len(word_set))

    print("分词后最小长度", min(sentence_len_list))
    print("分词后中间长度", sorted(sentence_len_list)[len(sentence_len_list) // 2])
    print("分词后最大长度", max(sentence_len_list))

    vcab_size = len(word_set)
    max_length = max(sentence_len_list)
    return data_df, word_set, vcab_size, max_length


def draw_history(history):
    # 绘制训练过程精度和损失曲线图
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def generate_data(data_df, max_length):
    generate_data = np.zeros((len(data_df["Sentence"]), max_length))
    for i, sentence in enumerate(data_df["Sentence"]):
        for j, word in enumerate(sentence):
            # index = word2index.get(word, -1)
            # if index != -1:
            #     generate_data[i, j, index] = 1.
            if j >= max_length:
                continue
            index = word2index.get(word, 0)
            if index != 0:
                generate_data[i, j] = index
    return generate_data


if __name__ == '__main__':
    train_data_df, word_set, vcab_size, max_length = prapare_without_tokenize(train_dev_data_path + "cn_train.csv")
    print("max_length: ", max_length)
    print("vcab_size: ", vcab_size)
    word_set = list(word_set)
    # 索引从1开始
    word_index = 1
    word2index = {}
    for word in word_set:
        word2index[word] = word_index
        word_index += 1

    train_generate_data = generate_data(train_data_df, max_length)
    # for sentence in data_df["Sentence"]:
    #     indices = []
    #     for word in sentence:
    #         indices.append(word2index.get(word, default=0))
    #     one_hot_code = tf.one_hot(indices=indices, depth=vcab_size)
    train_lables = np.array(train_data_df["Label"])
    count_result = Counter(train_lables)
    print("幽默：", count_result[1], " , 占比：", count_result[1] / len(train_lables))

    model = keras.Sequential()
    model.add(Embedding(vcab_size + 1, 512, input_length=max_length, mask_zero=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    # test_data_df, _, _, _ = prapare_without_tokenize(train_dev_data_path + "cn_dev.csv")
    # test_generate_data = generate_data(test_data_df, max_length)

    validation_metrics = Validation_Metrics()
    train_data_x, validate_data_x, train_data_y, validate_data_y = train_test_split(train_generate_data, train_lables,
                                                                                    test_size=0.2, random_state=24)
    history = model.fit(train_data_x, train_data_y, epochs=30, batch_size=512, callbacks=[validation_metrics],
                        validation_data=(validate_data_x, validate_data_y))

    draw_history(history)
