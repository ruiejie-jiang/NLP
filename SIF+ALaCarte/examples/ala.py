import numpy as np
import pandas as pd
import sys
sys.path.append('../src')
import data_io, params, SIF_embedding
from sklearn.linear_model import LinearRegression
from unicodedata import category

def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x the index for each words in each sentence
    """
    seq1 = []
    for i in sentences:
        set2idx = getSeq(i,words)
        if len(words) - 1 not in set2idx:
            seq1.append(getSeq(i,words))
    return seq1

def lookupIDX(words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    else:
        return len(words) - 1

def getSeq(p1,words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    return X1



def get_matrix(We, sentences, words, win=10):
    '''

    :param We: the word embedding for all words
    :param new_text_list: the sentences
    :param words: the index for words
    :param win: window size
    :return: a linear-regression model which is used to make prediction for rare-word
    '''
    x = sentences2idx(sentences, words)#the index for each word in each sentence
    average_emb = np.zeros((len(We), len(We[0])))#an array used to store context embedding
    numbers = np.zeros(len(We))#used to store the time of the word appeared in the context

    k = 0
    for i in x:
        k = k + 1
        for j in range(len(i)):
            if j > win-1 and j < len(i) - win:
                num = 2 * win
                average_emb[i[j]] = (np.sum(We[np.array(i[j - win:j + win+1])], axis=0) - We[i[j]]) / num
                numbers[i[j]] += 1
            elif j > win-1:
                num = len(i) - j + win - 1
                average_emb[i[j]] = (np.sum(We[np.array(i[j - win:])], axis=0) - We[i[j]]) / num
                numbers[i[j]] += 1
            elif j < len(i) - win and j < win:
                num = win + j
                average_emb[i[j]] = (np.sum(We[np.array(i[:j + win + 1])], axis=0) - We[i[j]]) / num
                numbers[i[j]] += 1
            else:
                num = len(i)
                average_emb[i[j]] = (np.sum(We[np.array(i[:])], axis=0) - We[i[j]]) / num
                numbers[i[j]] += 1

    index = np.where(numbers != 0)#the index of the word appeared
    numbers_true = numbers[index]
    numbers_true = numbers_true.reshape(len(numbers_true), 1)
    average_emb_true = average_emb[index]
    average_emb_true = average_emb_true / numbers_true#the context embedding

    We_true = We[index]#the true embedding

    reg = LinearRegression().fit(average_emb_true, We_true)#linear-regression model

    return reg





