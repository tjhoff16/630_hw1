import csv, math, sys
import numpy as np
from numpy import exp
import string
from sklearn.metrics import f1_score
import re
from collections import Counter
import pandas
from scipy.sparse import csr_matrix

def open_tsv(fl):
    rdata=[]
    f= open (fl, 'r', encoding='utf-8')
    for r in f.readlines():
        rdata.append(re.split('\t', r.replace('\n', '')))
    return rdata[1:100]

def tokenize (st):
    return re.split('\s+', st)

fl = open_tsv('train.tsv')

word_to_column={} #key,val == word, column
indptr = [0] # a big list of row, col, count
indices = [] # instance ids
data=[]
Y = []
vocab = {}
test = ["hello world hello", "goodbye cruel world"]
for ID,d,clss in fl:
    Y.append(clss)
    tokens = tokenize(d)
    for term in tokens:
        index = vocab.setdefault(term, len(vocab))
        indices.append(index)
        data.append(1)
        # print (index, indices, data)
    indptr.append(len(indices))
Y = np.array(Y)
    # print (indptr)
    # token_counts = Counter(tokens)
    # for token, count in token_counts.items():
    #     # col = word_to_column.get(token, (len(word_to_column)+1))
    #     if token in word_to_column:
    #         col = word_to_column[token]
    #     else:
    #         col = len(word_to_column)
    #         word_to_column[token] = col
    #     A.append((int(ID), col, count))

matrix = csr_matrix((data, indices, indptr), dtype=int)
# print (matrix.toarray(), matrix.shape)
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

def sigmoid(z):
    # array = []
    # print (z.shape)
    # for element in z:
    #     # print (element[2])
    #     array.append(float(1.0 / float((1.0 + exp(-1.0*element[2])))))
    # return np.array(array)
    return float(1.0 / float((1.0 + np.exp(-1.0*z))))

def logistic_regression(features, target, steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        print (intercept.shape, features.shape)
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])
    # print (features.shape, weights.shape)
    for step in np.arange(steps):
        scores = np.dot(features, weights)
        # print (scores.shape)
        predictions = sigmoid(scores)
        # print (scores, predictions)
        break

        # Update weights with log likelihood gradient
        output_error_signal = target - predictions

        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            print (log_likelihood(features, target, weights))

    return weights

weights = logistic_regression(matrix, Y, steps=300000, learning_rate=5e-5, add_intercept=True)
