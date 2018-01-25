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
# indptr = [0] # a big list of row, col, count
rows = []
cols = []
counts = []
Y = []
test = ["hello world hello", "goodbye cruel world"]
for row,(ID,text,clss) in enumerate(fl):
    Y.append(int(clss))
    tokens=tokenize(text)
    token_counts = Counter(tokens)
    for token, count in token_counts.items():
        if token in word_to_column:
            col = word_to_column[token]
        else:
            col = len(word_to_column)
            word_to_column[token] = col
        rows.append(row)
        cols.append(col)
        counts.append(count)

matrix = csr_matrix((counts, (rows, cols)), dtype=int)
def log_likelihood(features, target, weights):
    scores = np.dot(weights, features)
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
# print (matrix.shape[0])
# print (log_likelihood(matrix, Y, np.zeros(matrix.shape[0])))

def logistic_regression(features, target, steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((1, features.shape[0]))
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
