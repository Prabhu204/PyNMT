# -*- coding: utf-8 -*-
# @Author : Prabhu Appalapuri<prabhu.appalapuri@gmail.com>
# @Time : 20.02.20 16:27

from torch.autograd import Variable
from src.preprocess import *
from src.encoder import Encoder
from src.decoder import Decoder
from torch import nn
import argparse


def get_args():
    """

    :return: dict of input parameters
    """
    parser = argparse.ArgumentParser("""Evaluation of neural language translator model""")
    parser.add_argument("-l", "--layers", type=int, default= 2)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-n", "--num_epochs", type= int, default=20)
    parser.add_argument("-lr", "--lr", type=float, default= 0.01)
    parser.add_argument("-hs", "--hidden_size", type=int, default=440)
    parser.add_argument("-d", "--dropout", type=float, default=0.2)
    parser.add_argument("-encpath", "--encoder_path", type=str, default="models/English_model_enc_weights.pt")
    parser.add_argument("-decpath", "--decoder_path", type=str, default="models/German_model_dec_weights.pt")
    args = parser.parse_args()
    return args


def preprocess(input_sentence, en_vocab, en_word2index, output_sentence = None, de_vocab= None):
    """
    :param input_sentence: test sample of input language
    :param en_vocab: English language vocabulary
    :param en_word2index: word-to-index dictionary
    :param output_sentence: target language sentence for comparision
    :param de_vocab: target language vocabulary
    :return: tensored version of a indexed-input sentence
    """
    input_sentence = [decontracted(input_sentence.lower())]
    # print(input_sentence)
    sent2idx = convert_sent2index(input_sentence, en_word2index, en_vocab)
    # print(sent2idx)
    sent2idx = assign_sos_eos(lan=sent2idx, w2i=en_word2idx, sentlength=60, padding=True)
    input_variable = create_tensored_sentences(indexed_sentences=sent2idx)
    return input_variable


def evaluate(input_variable, encoder, decoder, word2index,index2word):
    """
    :param input_variable: tensored version of a indexed-input sentence
    :param encoder: input language encoder
    :param decoder: target language decoder
    :param word2index: target language word to index
    :param index2word: target language index to word of its vocabulary
    :return: translated version of input language to target language
    """
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_variable = input_variable.view(-1, 1)
        # print(input_variable.shape)
        enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(input_variable.shape[1])
        enc_hiddens, enc_outputs = encoder(input_variable, enc_h_hidden, enc_c_hidden)
        decoder_input = Variable(torch.LongTensor(1,1).fill_(word2index.get("SOS")))
        dec_h_hidden = enc_outputs[0]
        dec_c_hidden = enc_outputs[1]
        decoded_words = []
        for di in range(input_variable.shape[0]):
            pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)
            # print(f"Pred:\n{pred.shape}")
            # print(f"Pred:\n{pred}")
            topv, topi = pred.topk(1, dim=1)
            idx = topi.item()
            # print(pred[0][3])
            # print(f"topv:\n{topv}")
            # print(f"topi:\n{topi.item()}")
            if idx == word2index.get("EOS"):
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(index2word.get(int(idx)))
            decoder_input = Variable(torch.LongTensor(1, 1).fill_(idx))
            dec_h_hidden = dec_outputs[0]
            dec_c_hidden = dec_outputs[1]
        output_sentence = ' '.join(decoded_words)
        return output_sentence


if __name__ == '__main__':

    opt = get_args()
    bidirectional = False
    if bidirectional:
        directions = 2
    else:
        directions = 1
    en_vocabulary = load_data("EN_vocabulary")
    en_idx2word = load_data("EN_idx2word")
    en_word2idx = load_data("EN_word2idx")
    de_vocabulary = load_data("DE_vocabulary")
    de_idx2word = load_data("DE_idx2word")
    de_word2idx = load_data("DE_word2idx")

    input_sent = preprocess("five minutes later the first mountain-bikers set off.",
                            en_vocab=en_vocabulary,
                            en_word2index=en_word2idx)

    encoder_ = Encoder(len(en_vocabulary), opt.hidden_size, layers=opt.layers, bidirectional=bidirectional)
    decoder_ = Decoder(opt.hidden_size,
                       len(de_vocabulary),
                       layers=opt.layers,
                       dropout=opt.dropout,
                       bidirectional=bidirectional)
    encoder_.load_state_dict(torch.load(opt.encoder_path))
    decoder_.load_state_dict(torch.load(opt.decoder_path))
    pred_sent = evaluate(input_variable=input_sent,
                         encoder=encoder_,
                         decoder=decoder_,
                         word2index=de_word2idx,
                         index2word=de_idx2word)
    print(pred_sent)