import csv, math, sys, plotly, string, re
import numpy as np
from sklearn.metrics import f1_score

class TSV_row():
    def __init__(self, row, bt=False):
        self.ID = row[0]
        try:
            self.classifier = row[2]
        except:
            self.classifier=None
        self.tokens = tokenize(row[1])
        if bt:
            self.tokens = better_tokenize(row[1])

    def contains(self, inp):
        return inp==self.ID

def open_tsv(fl):
    rdata=[]
    f= open (fl, 'r', encoding='utf-8')
    for r in f.readlines(): rdata.append(re.split('\t', r.replace('\n', '')))
    print (len(rdata))
    return rdata

def tokenize (st):
    return re.split('\s+', st)

def generate_list_of_instances(tsv, better_tokenize=False):
    return [TSV_row(x, better_tokenize) for x in tsv][1:]

def train(tweets, smoothing_alpha=0):
    tot, hate_words, not_hate_words, ct = len(tweets), {}, {}, 0
    for e in tweets:
        if e.classifier == '1':
            ct+=1
            for w in e.tokens: hate_words[w] = hate_words.get(w, 0)+1
        else:
            for w in e.tokens: not_hate_words[w] = not_hate_words.get(w, 0)+1
    nhct = sum(not_hate_words.values())
    hct = sum(hate_words.values())
    pH, pNH = ct/tot, (tot-ct)/tot
    total_words = len({**hate_words, **not_hate_words})
    print (total_words)
    return total_words, pH, pNH, hct, nhct, not_hate_words, hate_words, smoothing_alpha

def pWord(word, hate, smoothing_alpha):
    if hate: return np.log((hw.get(word,0)+smoothing_alpha)/(hct+smoothing_alpha*total_words))
    return np.log((nhw.get(word,0)+smoothing_alpha)/(nhct+smoothing_alpha*total_words))

def pTweet(message, hate, smoothing_alpha):
    r = 1.0
    for w in message: r += pWord(w, hate, smoothing_alpha)
    return r

def classify(message, pH, pNH, smoothing_alpha=0, testing=False):
    isHate = np.log(pH)+pTweet(message.tokens, True, smoothing_alpha)
    notHate = np.log(pNH)+pTweet(message.tokens, False, smoothing_alpha)
    if testing:
        if isHate > notHate: cs = '1'
        else: cs = '0'
        if cs == message.classifier: res = 'match'
        else: res = 'not match'
        return message.ID, cs, message.classifier, res
    else: return isHate > notHate

def better_tokenize(st):
    with open('stopwords', 'r') as f: stwds = f.read().split('\n')
    stopwords=[re.sub('[^\w\s\d]','',s.lower()) for s in stwds]
    exclude = set(string.punctuation)
    st=st.lower()
    st= set(re.split('\s+',''.join(ch for ch in st if ch not in exclude)))
    return [w for w in st if w not in stopwords]

fl = open_tsv('train.tsv')
wds = generate_list_of_instances(fl, better_tokenize=True)
ffl = open_tsv('dev.tsv')
tts = generate_list_of_instances(ffl, better_tokenize=True)
rng=.6
total_words, pH, pNH, hct, nhct, nhw, hw, sma = train(wds, smoothing_alpha=rng)
y_true=[]
y_pred=[]
tot_right=0
tt=0
for e in tts:
    x = np.array(classify(e, pH, pNH, smoothing_alpha=rng, testing=True))
    y_true.append(x[1])
    tt+=1
    if x[3]=='match':
        tot_right+=1
        y_pred.append(x[1])
    else:
        if x[1] == '1': y_pred.append(0)
        else: y_pred.append(1)
score = f1_score(y_true, y_pred, average='weighted')
print (score, rng, total_words, tot_right)

val = max(hw.values())
for k,v in hw.items():
    if v == val:
        print (k, v)



#
# def sigmoid(X):
#     '''Compute the sigmoid function '''
#     #d = zeros(shape=(X.shape))
#     den = 1.0 + e ** (-1.0 * X)
#     d = 1.0 / den
#     return d
#
# def compute_cost(theta,X,y): #computes cost given predicted and actual values
#     m = X.shape[0] #number of training examples
#     theta = reshape(theta,(len(theta),1))
#     #y = reshape(y,(len(y),1))
#     J = (1./m) * (-transpose(y).dot(log(sigmoid(X.dot(theta)))) - transpose(1-y).dot(log(1-sigmoid(X.dot(theta)))))
#     grad = transpose((1./m)*transpose(sigmoid(X.dot(theta)) - y).dot(X))
#     #optimize.fmin expects a single value, so cannot return grad
#     return J[0][0]#,grad
#
# def compute_grad(theta, X, y):
#     #print theta.shape
#     theta.shape = (1, 3)
#     grad = zeros(3)
#     h = sigmoid(X.dot(theta.T))
#     delta = h - y
#     l = grad.size
#     for i in range(l):
#         sumdelta = delta.T.dot(X[:, i])
#         grad[i] = (1.0 / m) * sumdelta * - 1
#     theta.shape = (3,)
#     return grad
#
# def predict(theta, X):
#     '''Predict whether the label
#     is 0 or 1 using learned logistic
#     regression parameters '''
#     m, n = X.shape
#     p = zeros(shape=(m, 1))
#     h = sigmoid(X.dot(theta.T))
#     for it in range(0, h.shape[0]):
#         if h[it] > 0.5:
#             p[it, 0] = 1
#         else:
#             p[it, 0] = 0
#     return p
# # #Compute accuracy on our training set
# # p = predict(array(theta), it)
# # print 'Train Accuracy: %f' % ((y[where(p == y)].size / float(y.size)) * 100.0)
