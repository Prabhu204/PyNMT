# -*- coding: utf-8 -*-
# @Author : Prabhu Appalapuri<prabhu.appalapuri@gmail.com>
# @Time : 20.02.20 10:07

import glob
import torch
import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
import re
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def get_file_names(fileSuffix):
    """
    :param fileSuffix: language code (en --> English, de-->German)
    :return: list of file names
    """
    files = []
    for file in glob.glob("English_German_news_pairs/*.{}".format(fileSuffix)):
        files.append(file)
    return files


def decontracted(phrase):
    """
    :param phrase: sentence
    :return: decontracted version of a input sentence
    """
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def get_sentences(file):
    """
    :param file: file path
    :return: list of sentences without shuffling.
    """
    sentences = []
    with open(file, 'r') as f:
        for sent in f:
            sent = decontracted(sent.lower())
            sentences.append(sent)
    return sentences


def word_count(sentences):
    """
    :param sentences: list of sentences
    :return: dictionary of each word count
    """
    wordCount = Counter(word.strip(',." ;:)(][?!') for sent in sentences for word in sent.split())
    return wordCount


def vocabulary(word_counts):
    """
    :param word_counts: dictionary of each word count
    :return: list of vocabulary
    """
    vocabulary = list(map(lambda x: x[0], sorted(word_counts.items(), key=lambda x: -x[1])))
    return vocabulary


def get_word2index(vocab):
    """
    :param vocab: list of vocabulary
    :return: dictionary of word to index
    """
    start_idx = 4
    word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(vocab)])
    word2idx['SOS'] = 0
    word2idx['EOS'] = 1
    word2idx["<UNK>"] = 2
    word2idx["<PAD>"] = 3
    return word2idx


def get_index2word(word2idx):
    """
    :param word2idx: dictionary of word to index
    :return: reverse order of input dictionary
    """
    idx2word = dict([(idx, word) for word, idx in word2idx.items()])
    return idx2word


def convert_sent2index(sentences, word2idx, vocab):
    """
    :param sentences: list of sentences
    :param word2idx: dictionary of word to index
    :param vocab: vocabulary of given language
    :return: converted version entire sentence into index based sentence
    """
    sent2idx = [[word2idx.get(word.strip(',." ;:)(][?!')) if word.strip(',." ;:)(][?!') in vocab else word2idx.get("<UNK>") for word in sentence.split()]
                for sentence in sentences]
    return sent2idx


def show_plot(english, german, filename):
    """
    :param english: list of indexed sentences of english language
    :param german: list of indexed sentences of german language
    :param filename: saving a figure with a name
    :return: location of saved figure directory
    """
    en_len = [len(item) for item in english]
    de_len = [len(item) for item in german]

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    ax1 = sns.distplot(en_len, ax=axs[0], axlabel='EN Sentences')
    ax2 = sns.distplot(de_len, ax=axs[1], axlabel='DE Sentences')
    plt.savefig(f"results/Sent_{filename}_trimming.png")
    return "Figure has saved to <results> directory"


def apply_threshold(en, de, sentlength):
    """
    :param en: English language sentences
    :param de: German language sentences
    :param sentlength: cutoff length of a sentences
    :return: trimmed sentences
    """
    en_sents= []
    de_sents = []
    for i in range(len(en)):
        en_len = len(en[i])
        de_len = len(de[i])
        if en_len < de_len:
            n = en_len
        else:
            n = de_len
        if abs(en_len-de_len) <=(0.3*n):
            if en_len<= sentlength and de_len<=sentlength:
                en_sents.append(en[i])
                de_sents.append(de[i])
    return en_sents, de_sents


def assign_sos_eos(lan, sentlength, w2i, padding=True):
    """
    :param lan: list of indexed-sentences
    :param sentlength: cutoff length of sentence
    :param w2i: dictionary of word to index
    :param padding: applying padding to a end of a sentence
    :return: padded sentences
    """
    if padding:
        for i in range(len(lan)):
            lan[i] = [w2i['SOS']] + lan[i] + [w2i["EOS"]] + (sentlength - len(lan[i])) * [w2i['<PAD>']]
            # de[i] = [de_w2i['SOS']] + de[i] + [de_w2i["EOS"]] + (sentlength - len(de[i])) * [de_w2i['<PAD>']]
    else:
        for i in range(len(lan)):
            lan[i] = [w2i['SOS']] + lan[i] + [w2i["EOS"]]
            # de[i] = [de_w2i['SOS']]+de[i]+[de_w2i["EOS"]]
    return lan


def onehot_encode(sent_indexes, vocab):
    """
    :param sent_indexes:
    :param vocab:
    :return:
    """
    onehotEncoding = []
    for sent in sent_indexes:
        vocabArray = [0 for _ in range(len(vocab))]+ 4*[0]
        for idx in sent:
            vocabArray[idx]=1
        onehotEncoding.append(vocabArray)
    return onehotEncoding


def saving_preprocessed_data(data, file_name):
    """
    :param data: variable to save
    :param file_name: name of a file to save a variable
    :return: file saved
    """
    try:
        with open(f'data/{file_name}.pickle', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print("Something is worng with file saving!!!!!!")
    return 'file saved'


def load_data(file_name):
    with open(f'data/{file_name}.pickle', 'rb') as handle:
            data = pickle.load(handle)
    return data


def data_splitting(english, german, test_size):
    """
    :param english: input language
    :param german: taget language
    :param test_size: testset size
    :return: train & test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(english, german, test_size=test_size)
    return X_train, X_test, y_train, y_test


def create_tensored_sentences(indexed_sentences):
    """
    :param indexed_sentences: indexed version of a sentence
    :return: tensor of a indexed sentence
    """
    result = np.array(indexed_sentences)
    # print(result.shape)
    result = torch.LongTensor(result)
    return result


def generate_batches(input_language, output_language, batch_size, shuffle = False):
    """
    :param input_language:  input_language
    :param output_language: output_language
    :param batch_size: batch_size
    :param shuffle: shuffling
    :return: batches of dataset
    """
    input_tensors = create_tensored_sentences(indexed_sentences=input_language)
    output_tensors = create_tensored_sentences(indexed_sentences = output_language)
    input_genarator = DataLoader(input_tensors, shuffle=shuffle, batch_size=batch_size)
    output_genarator = DataLoader(output_tensors, shuffle=shuffle, batch_size=batch_size)
    return input_genarator, output_genarator
