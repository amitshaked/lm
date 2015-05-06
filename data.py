#!/usr/bin/env python2.7
import re
import string
from collections import defaultdict
from collections import namedtuple
from progressbar import ProgressBar
from hashlib import sha1
from math import log, exp
import os
import sys

LOGZERO = -sys.maxint - 1

ALPHABET = [chr(ord('a') + i) for i in xrange(ord('z') - ord('a') + 1)] + [chr(ord('A') + i) for i in xrange(ord('Z') - ord('A') + 1)]

def tokenize(s):
    def is_word(x):
        return len(x) and x[0] in ALPHABET
    def convert(x):
        return re.sub('[^a-zA-Z]', '', x).lower()
    return filter(is_word, map(convert, re.findall(r"[\w'.]+", s)))

def add_log(x, y):
    x,y = max(x,y), min(x,y)

    if y <= LOGZERO:
        return x

    negdiff = y-x
    return x + log(1 + exp(negdiff))

class LanguageModel(object):
    def __init__(self, n, lines = None, voc = None, smoothing = None, lmbd = None):        
        self.smoothing = smoothing
        self.lmbd = lmbd
        self.models = defaultdict(dict)
        self.voc = []        
        if lines == None:
            self.voc = []
            if voc != None:
                self.voc = voc
            self.n = n            
            return

        lines = lines[1:] # /data/

        # read total ngrams num:
        i = 0
        for i in xrange(len(lines)):
            if lines[i].startswith('\\'):
                break
            if lines[i] == '\n':
                continue
            a = lines[i].strip().split('=')
            self.models[int(a[0][6])] = {}
            self.n = int(a[0][6])            

        lines = lines[i-1:]
        #read ngrams
        for j in xrange(len(self.models)):
            if len(lines) == 0:
                break
            if lines[0] == '\n':
                break
            n = int(lines[0][1]) # \n-grams:
            k = 1
            for k in xrange(1, len(self.models[n]) + 1):
                a = lines[k].split('\t')
                words = a[1].split()
                self.voc.extend(words)
                self.models[n][tuple(words)] = float(a[0]) #ngrams[words] = prob
                
            lines = lines[k:]

        self.voc = list(set(self.voc))

    def set_model(self, n, model):
        self.models[n] = model

    def setn(self, n):
        self.n = n

    def set_voc(self, voc):
        self.voc = voc

    def set_prob(self, n, words, log_prob):
        w = tuple(words)
        self.models[n][w] = log_prob

    def set_smoothing(self, s, lmbd):
        self.smoothing = s
        self.lmbd = lmbd

    def get_smoothing(self):
        return self.smoothing, self.lmbd

    def get_prob(self, words):
        words = tuple(words)
        d = self.models[len(words)]
        if words not in d:
            return -20  # FIXME: Return <UNK> prob
        else:
            return d[words]

    def __getitem__(self, item):
        return self.models[item]

    def __len__(self):
        return len(self.models)

    def dump(self, output_file):
        with open(output_file, 'wb') as f:
            f.write('\\data\\\n')
            for n in self.models:
                f.write('ngram %d=%d\n' % (n, len(self.models[n])))
            f.write('\n')
            f.write('\\smoothing\\\n')
            f.write('%s\t%.32f\n' % (self.smoothing, self.lmbd))
            f.write('\n')
            for n in self.models:
                f.write('\\%d-grams:\n' % n)    
                for words, prob in self.models[n].iteritems():
                    f.write('%.32f\t%s\n' % (prob ," ".join(words)))
    
    @staticmethod
    def load(f):
        f.readline()  # \data\
        ngram = 0
        while True:
            l = f.readline().strip()
            if not l:
                break
            ngram = max(ngram, int(l.split('ngram ')[1].split('=')[0]))
        
        assert ngram != 0, "Can't find ngram in file!"
        
        lm = LanguageModel(ngram)
        
        f.readline() # \smoothing\
        smooth_line = f.readline().strip()
        f.readline()
        
        smoothing, lmbd = smooth_line.split()
        lm.set_smoothing(smoothing, float(lmbd))
        
        current_ngram = 0
        voc = set()
        while True:
            l = f.readline().strip()
            if not l:
                break
            elif l.startswith('\\'):  # descriptor
                current_ngram = int(l[1])
            else:  # data line
                assert current_ngram != 0, 'Invalid n-gram'
                log_prob, words = l.split('\t', 1)
                log_prob = float(log_prob)
                words = tuple(words.split(' '))
                lm.set_prob(current_ngram, words, log_prob)
                if current_ngram == 1:
                    voc.add(words[0])
        
        lm.set_voc(voc)
        
        return lm

    def smooth(self):
        if self.smoothing == 'ls':
            self._smooth_ls()
        elif self.smoothing == 'wb':
            raise NotImplementedError('Witten-Bell smoothing')
        elif self.smoothing == 'none':
            pass
        else:
            raise ValueError('Invalid smoothing: %s' % self.smoothing)

    def _smooth_ls(self):
        pass

def load_test_file(n, lines):
    n_grams = []
    for line in lines:
        try:
            l = ['<s>'] + tokenize(line) + ['</s>']
        except:
            continue
        n_grams.extend((l[max(0, x-n+1):x+1] for x in xrange(len(l))))
    return n_grams


def calculate_prob_ls(n, n_grams, m_grams, gram, size, smoothing, lmbd):
    try:
        a = log(n_grams[gram] + lmbd*size) - log(m_grams[gram[:-1]] + size)
    except:
        print gram, size, n_grams[gram], m_grams[gram[:-1]]
        return 0
    assert m_grams[gram[:-1]] >= n_grams[gram]
    return a

def build_model(n, lines):
    m_grams = dict()
    lm = LanguageModel(n)
    
    print 'Building model...'
        
    progress = ProgressBar(maxval=len(lines)).start()            
    grams = [defaultdict(int) for _ in xrange(n+1)]
    voc=set()
    tokens = 0
    for i in xrange(len(lines)):            
        words = ['<s>'] + tokenize(lines[i]) + ['</s>']       
        for word in words:
            voc.add(word)
        num_words = len(words)
        tokens += num_words
        
        for l in xrange(1, n+1):
            for j in xrange (l, num_words+1):            
                a = tuple(words[j-l:j])
                grams[l][a] +=1                
        progress.update(i)        
    progress.finish()
            
    grams[0] = dict()
    grams[0][tuple()] = tokens
    voc = list(set(voc))
    lm.set_voc(voc)
    
    for l in xrange(1, n+1):
        for gram, gram_count in grams[l].iteritems():
            log_prob = log(gram_count) - log(grams[l-1][gram[:-1]])
            lm.set_prob(l, gram, log_prob)

    return lm

def load_file(path, n = 2):
    filename = os.path.split(path)[1]    
    if filename.endswith('lm'):
        loader, typename = LanguageModel.load, 'Language model'
    elif filename.endswith('corp'):
        loader, typename = build_model, 'Corp file'
    elif filename.endswith('test'):
        loader, typename = load_test_file, 'Test file'   
    else:
        raise NotImplementedError('Unknown file type')

    with open(path, 'rb') as f:
        lines = f.readlines()
        print 'Loading %s %s...' % (typename, filename)
        ret = loader(n, lines)
        return ret


def normalize(a):   
    s = float(sum(a))
    for i in a:
        a[i] /= s