# -*- coding: utf-8 -*-
# @Author : Prabhu Appalapuri<prabhu.appalapuri@gmail.com>
# @Time : 20.02.20 09:49

import pandas as pd
from torch import nn
from torch import optim
from torch.autograd import Variable
from src.preprocess import *
from src.encoder import Encoder
from src.decoder import Decoder
import argparse

bidirectional = False
if bidirectional:
    directions = 2
else:
    directions = 1


def get_args():
    """
    :return: dict of input parameters
    """
    parser = argparse.ArgumentParser("""Implementation of neural language translator from English to German""")
    parser.add_argument("-l", "--layers", type=int, default= 2)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-n", "--num_epochs", type= int, default=20)
    parser.add_argument("-lr", "--lr", type=float, default= 0.01)
    parser.add_argument("-hs", "--hidden_size", type=int, default=440)
    parser.add_argument("-d", "--dropout", type=float, default=0.2)
    parser.add_argument("-bi", "--bidirectional", action="store_true", default=True)
    args = parser.parse_args()
    return args


def test(en_model, de_model, test_input, test_target, word2index):
    """
    :param en_model: input language encoder
    :param de_model: target language decoder
    :param test_input: testset of input language
    :param test_target: testset of target language
    :param word2index: target language word 2 index dictionary
    :return: average loss value of target language(i.e testset)
    """
    criterion = nn.NLLLoss()
    de_model.eval()
    en_model.eval()
    test_target = list(test_target)
    with torch.no_grad():
        test_loss = 0
        for idx, batch in enumerate(test_input):
            target_batch = test_target[idx]
            loss = 0
            # create initial hidden state for encoder
            enc_h_hidden, enc_c_hidden = en_model.create_init_hiddens(batch.shape[1])
            enc_hiddens, enc_outputs = en_model(batch, enc_h_hidden, enc_c_hidden)
            decoder_input = Variable(torch.LongTensor(1, batch.shape[1]).fill_(word2index.get("SOS")))

            dec_h_hidden = enc_outputs[0]
            dec_c_hidden = enc_outputs[1]
            for i in range(target_batch.shape[0]):
                pred, dec_outputs = de_model(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)
                topv, topi = pred.topk(1, dim=1)
                ni = topi.view(1, -1)
                decoder_input = ni
                dec_h_hidden = dec_outputs[0]
                dec_c_hidden = dec_outputs[1]
                loss += criterion(pred, target_batch[i])
            test_loss += loss.item() / target_batch.shape[0]
    return test_loss /len(test_input)


def save_model(avg_epoch_loss, encoder, decoder, encoder_name, decoder_name):
    """
    :param avg_epoch_loss: Loss value of a testset per epoch
    :param encoder: input language encoder
    :param decoder: taget language decoder
    :param encoder_name: to save a trained encoder with a name
    :param decoder_name: to save a trained decoder with a name
    :return: lowest loss achieved on testset in an epoch
    """
    lowest_loss = 100.00
    if avg_epoch_loss < lowest_loss:
        print(avg_epoch_loss)
        lowest_loss = avg_epoch_loss
        torch.save(encoder.state_dict(), "models/"+encoder_name+'_enc_weights.pt')
        torch.save(decoder.state_dict(), "models/"+decoder_name+'_dec_weights.pt')
    return lowest_loss


def train(opt, input_data, output_data, optimizer, en_vocab, de_vocab, test_input, test_target, word2index):
    """
    :param opt: dictionary of input parameters to train encoder & decoder
    :param input_data: input language trainset
    :param output_data: target language in trainset
    :param optimizer: SGD or ADAM
    :param en_vocab: English language vocabulary from an entire available dataset
    :param de_vocab: German language vocabulary from an entire available dataset
    :param test_input: input language testset
    :param test_target: target language in testset
    :param word2index: target language word 2 index dictionary
    :return: training of encoder and decoder is done!!
    """
    encoder = Encoder(len(en_vocab), opt.hidden_size, layers=opt.layers, bidirectional=bidirectional)
    decoder = Decoder(opt.hidden_size, len(de_vocab), layers=opt.layers, dropout=opt.dropout,
                      bidirectional=bidirectional)
    if optimizer == "sgd":
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=opt.lr, momentum=0.9)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=opt.lr)
    else :
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr)

    criterion = nn.NLLLoss()
    
    encoder.train()
    decoder.train()
    
    number_iters = len(input_data)
    output_data = list(output_data)
    # loss = 0
    count = 0
    for epoch in range(opt.num_epochs):
        epoch_loss = 0
        for idx, batch in enumerate(input_data):
            loss = 0
            output_batch = output_data[idx]
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
#             print(batch)
            
            enc_h_hidden, enc_c_hidden = encoder.create_init_hiddens(batch.shape[1])
#             print(enc_c_hidden)
#             enc_h_hidden = enc_h_hidden.cuda()
#             enc_c_hidden = enc_c_hidden.cuda()
            enc_hiddens, enc_outputs = encoder(batch, enc_h_hidden, enc_c_hidden)
            
            # enc_hiddens = enc_hiddens.cuda()
            # enc_outputs = enc_outputs.cuda()
#             print(f'Encoder_h_hidden:\n{enc_hiddens.shape}\n***********')
#             print(f'Encoder Outputs:\n{enc_outputs}\n***********')

            decoder_input = Variable(torch.LongTensor(1,batch.shape[1]).fill_(word2index.get("SOS")))
            
#             print(f'decoder_input shape:\n{decoder_input.shape}\n***********')
#             print(f'decoder_input :\n{decoder_input}\n***********')
            
            dec_h_hidden = enc_outputs[0]
            dec_c_hidden = enc_outputs[1]
#             print(f'dec_h_hidden shape:\n{dec_h_hidden.shape}\n***********')
#             print(f'dec_h_hidden :\n{dec_h_hidden}\n***********')
#             print(f'dec_c_hidden shape:\n{dec_c_hidden.shape}\n***********')
#             print(f'dec_c_hidden :\n{dec_c_hidden}\n***********')
            
            for i in range(output_batch.shape[0]):
                pred, dec_outputs = decoder(decoder_input, dec_h_hidden, dec_c_hidden, enc_hiddens)
                
                decoder_input = output_batch[i].view(1,-1)
                
#                 print(f'decoder_input shape :\n{decoder_input.shape}\n***********')
#                 print(f'decoder_input :\n{decoder_input}\n***********')
                
                dec_h_hidden = dec_outputs[0]
                dec_c_hidden = dec_outputs[1]
                # print(pred)
                # print(output_batch[i])
                loss += criterion(pred, output_batch[i])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(),0.25)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),0.25)

            encoder_optimizer.step()
            decoder_optimizer.step()
            
            avg_iter_loss = loss.item() / output_batch.shape[0]
            epoch_loss += avg_iter_loss
            print(f"Epoch:[{epoch+1}/{opt.num_epochs}]\tIter:[{idx+1}/{number_iters}]\tIter Loss:{avg_iter_loss}")
        avg_epoch_loss = epoch_loss/number_iters

        with open("results/Train_result.txt", "a") as f:
            if epoch == 0:
                f.write(f"Bidirectional:\t{bidirectional}\nLayers:\t{opt.layers}\nHidden_size:\t{opt.hidden_size}\n")
                f.write(f"Dropout:\t{opt.dropout}\nBatch_size:\t{opt.batch_size}\nEpochs:\t{opt.epochs}\nlearning_rate:\t{opt.lr}\n")
            f.write(f"Epoch:[{epoch+1}/{opt.num_epochs}]\t\tEpoch Loss:{avg_epoch_loss}\n")

        # test model for every epoch and save models for lesser models

        test_loss = test(en_model=encoder, de_model=decoder, test_input=test_input, test_target=test_target,
                         word2index=word2index)
        with open("results/Test_result.txt", "a") as f:
            f.write(f"Epoch:[{epoch+1}/{opt.num_epochs}]\t\tTest Loss:{test_loss}\n")

        # save trained model with reduced loss
        lowest_loss = save_model(avg_epoch_loss=test_loss, encoder=encoder, decoder=decoder,
                                 encoder_name="English_model", decoder_name="German_model")
        if epoch_loss == lowest_loss:
            if count <=5:
                count +=1
            else:
                break
        else:
            count=0
        encoder.train()
        decoder.train()
    return "Model traing has been done!!!!!"


if __name__ == '__main__':
    opt = get_args()
    en_files = get_file_names('en')
    de_files = get_file_names('de')
    en_files = sorted(en_files)
    de_files = sorted(de_files)
    en_sentences = [get_sentences(file) for file in en_files]
    print("EN sentences in a single file: ", len(en_sentences[0]))
    en_sentences = [item for sublist in en_sentences for item in sublist]
    print("Total EN sentences: ", len(en_sentences))
    de_sentences = [get_sentences(file) for file in de_files]
    print("DE sentences in a single file: ", len(de_sentences[0]))
    de_sentences = [item for sublist in de_sentences for item in sublist]
    print("Total DE sentences: ", len(de_sentences))

    df = pd.concat([pd.DataFrame(en_sentences), pd.DataFrame(de_sentences)], axis=1)
    df.columns = ["English", "German"]
    df.to_csv("data/english_german_pair.csv", index=False)

    en_wordCount = word_count(en_sentences)
    de_wordCount = word_count(de_sentences)
    en_vocabulary = vocabulary(en_wordCount)
    de_vocabulary = vocabulary(de_wordCount)

    print("EN vocabulary: ", len(en_vocabulary))
    print("DE vocabulary: ", len(de_vocabulary))

    en_word2idx = get_word2index(en_vocabulary)
    print(f'EN word to index: {list(en_word2idx.items())[:5]}')
    de_word2idx = get_word2index(de_vocabulary)
    print(f'DE word to index: {list(de_word2idx.items())[:5]}')

    en_vocabulary = en_vocabulary + ["SOS", "EOS", "<UNK>", "<PAD>"]
    de_vocabulary = de_vocabulary + ["SOS", "EOS", "<UNK>", "<PAD>"]

    en_sent2idx = convert_sent2index(en_sentences, en_word2idx, en_vocabulary)
    print(en_sent2idx[:2])
    print("*************")
    de_sent2idx = convert_sent2index(de_sentences, de_word2idx, de_vocabulary)
    print(de_sent2idx[:2])

    en_idx2word= get_index2word(en_word2idx)
    de_idx2word = get_index2word(de_word2idx)

    # Let's verify sentence encoding
    print("Encoded English sentence:\n", en_sent2idx[5000])
    print("***************")
    decode_en = " ".join([en_idx2word[idx] for idx in en_sent2idx[5000]])
    print("Decoded English sentence:\n", decode_en)

    print("Encoded German sentence:\n", de_sent2idx[5000])
    print("***************")
    decode_de = " ".join([de_idx2word[idx] for idx in de_sent2idx[5000]])
    print("Decoded German sentence:\n", decode_de)

    print(len(en_sent2idx))
    print(len(de_sent2idx))
    fig1 = show_plot(english=en_sent2idx, german=de_sent2idx, filename="before")
    en_sent2idx, de_sent2idx = apply_threshold(en_sent2idx, de_sent2idx, sentlength=60)
    print(len(en_sent2idx))
    print(len(de_sent2idx))
    fig2 = show_plot(english=en_sent2idx, german=de_sent2idx, filename="after")
    en_sent2idx= assign_sos_eos(lan = en_sent2idx, w2i=en_word2idx, sentlength=60, padding=True)
    de_sent2idx = assign_sos_eos(lan= de_sent2idx, w2i=de_word2idx, sentlength=60, padding=True)

    # saving_preprocessed_data(EN_idx2word, "EN_idx2word")
    # saving_preprocessed_data(DE_idx2word, "DE_idx2word")
    # saving_preprocessed_data(EN_vocabulary, "EN_vocabulary")
    # saving_preprocessed_data(DE_vocabulary, "DE_vocabulary")
    # saving_preprocessed_data(EN_wordCount, "EN_wordCount")
    # saving_preprocessed_data(EN_sent2idx, "EN_sent2idx")
    # saving_preprocessed_data(DE_wordCount, "DE_wordCount")
    # saving_preprocessed_data(DE_sent2idx, "DE_sent2idx")
    # saving_preprocessed_data(EN_word2idx, "EN_word2idx")
    # saving_preprocessed_data(DE_word2idx, "DE_word2idx")

    # EN_sent2idx = load_data("EN_sent2idx")
    # DE_sent2idx = load_data("DE_sent2idx")
    # EN_vocabulary = load_data( "EN_vocabulary")
    # DE_vocabulary = load_data("DE_vocabulary")

    X_train, X_test, y_train, y_test = train_test_split(en_sent2idx, de_sent2idx, test_size=0.1, random_state=25)

    input_data, output_data = generate_batches(input_language=X_train,
                                               output_language=y_train,
                                               batch_size=opt.batch_size)
    test_input_data, test_output_data = generate_batches(input_language=X_test,
                                                         output_language=y_train,
                                                         batch_size=opt.batch_size)
    training = train(opt= opt, input_data=input_data,
                     output_data=output_data,
                     optimizer='sgd',
                     en_vocab=en_vocabulary,
                     de_vocab=de_vocabulary,
                     test_input=test_input_data,
                     test_target=test_output_data,
                     word2index=de_word2idx)
    print(training)
