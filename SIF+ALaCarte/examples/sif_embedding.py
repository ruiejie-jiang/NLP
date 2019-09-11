import sys
sys.path.append('../src')
import data_io, params, SIF_embedding, ala
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm

# input
wordfile = '../data/glove.6B.300d.txt' # word vector file, can be downloaded from GloVe website
weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
#sentences = ['this is an example sentence asdascasx', 'this is also an example', 'this is another sentence that is slightly longer','hello my best friend']

x_train_df = pd.read_csv('../data/x_train.csv')
y_train_df = pd.read_csv('../data/y_train.csv')

tr_text_list = x_train_df['text'].values.tolist()
res_text_list = y_train_df['is_positive_sentiment'].values.tolist()
y_res = np.array(res_text_list)

new_text_list = list()

for text in tr_text_list:
    text = str.lower(text)
    temp = list()
    for j in text:
        if j.isalpha() or j == ' ' or j.isdigit():
            temp.append(j)
    temp = ''.join(temp)
    new_text_list.append(temp)

weight_dic = {}
N = 0
for i in new_text_list:
    for j in i.split():
        weight_dic[j] = weight_dic.get(j, 0) + 1
        N = N + 1
a=1e-3
for key, value in weight_dic.items():
    weight_dic[key] = a / (a + value / N)



# load word vectors
(words, We) = data_io.getWordmap(wordfile)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# load sentences


x, m = data_io.sentences2idx(new_text_list, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
w = data_io.seq2weight(x, m, weight4ind) # get word weights
win = 10
A = ala.get_matrix(We, new_text_list, words, win=10)


# set parameters
params = params.params()
params.rmpc = rmpc
# get SIF embedding
embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i

X_train, X_test, y_train, y_test = train_test_split(embedding, y_res, test_size=0.5, random_state=1)
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
b = clf.predict(X_test)
np.sum(np.abs(b-y_test))


def weighted_average_sim_rmpc(emb1, emb2):

    inn = (emb1 * emb2).sum()
    emb1norm = np.sqrt((emb1 * emb1).sum())
    emb2norm = np.sqrt((emb2 * emb2).sum())
    scores = inn / emb1norm / emb2norm
    return scores

