#!/usr/bin/env python2.7
import re
import string
from collections import defaultdict, namedtuple
from functools import partial
from progressbar import ProgressBar
from hashlib import sha1
from math import log10 as log
import os
import sys

LOGZERO = -sys.maxint - 1
exp = partial(pow, 10)

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
    def __init__(self, n):
        self.smoothing = None
        self.lmbd = 0
        self.models = defaultdict(dict)
        self.voc = []
        self.n = n
        self.prob_no_information = LOGZERO

    def set_model(self, n, model):
        self.models[n] = model

    def setn(self, n):
        self.n = n

    def set_voc(self, voc):
        self.voc = voc
        self.voc_size = len(voc)

    def set_prob(self, n, words, log_prob):
        w = tuple(words)
        self.models[n][w] = log_prob

    def set_smoothing(self, s, lmbd):
        self.smoothing = s
        if self.smoothing == 'ls':
            self.lmbd = lmbd

    def get_smoothing(self):
        return self.smoothing, self.lmbd

    def get_prob(self, words):
        n = len(words)
        words = tuple(w if w[0] == '<' or w in self.voc else '<UNK>' for w in words)
        d = self.models[n]
        if words in d:
            return d[words]

        if len(words) > 1 and words[:-1] not in self.models[n-1]:
            # Previous words hasn't been seen - use probability when there's no information
            return self.prob_no_information

        # Check whether we did some smoothing which raised nonzero probs to some value
        other = words[:-1] + ('<OTHER>',)
        if other in d:
            return d[other]

        # Shouldn't reach here!
        raise Exception(words)

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
            f.write('%.32f\n' % self.prob_no_information)
            f.write('\n')
            for n in self.models:
                f.write('\\%d-grams:\n' % n)
                for words, prob in self.models[n].iteritems():
                    f.write('%.32f\t%s\n' % (prob ," ".join(words)))

    @staticmethod
    def load(f):
        # \data\
        f.readline()
        ngram = 0
        while True:
            l = f.readline().strip()
            if not l:
                break
            ngram = max(ngram, int(l.split('ngram ')[1].split('=')[0]))

        assert ngram != 0, "Can't find ngram in file!"

        lm = LanguageModel(ngram)

        # \smoothing\
        f.readline()
        smooth_line = f.readline().strip()
        lm.prob_no_information = float(f.readline().strip())
        f.readline()

        smoothing, lmbd = smooth_line.split()
        lm.set_smoothing(smoothing, float(lmbd))

        # N-grams
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

    def train_model(self, lines, silent=False):
        if silent:
            def log(s):
                pass
        else:
            def log(s):
                print s

        log('Tokenizing...')
        lines_tokens = [['<s>'] + tokenize(l) + ['</s>'] for l in lines]

        # Build the lexicon as words that appear more than once, should be at least 99% of number of tokens
        log('Building lexicon...')
        lexicon = set()
        unk = set()
        counts = defaultdict(int)
        for l in lines_tokens:
            for w in l:
                if w[0] == '<':
                    continue

                counts[w] += 1

                if counts[w] == 1:
                    unk.add(w)
                elif counts[w] == 2:
                    lexicon.add(w)
                    unk.remove(w)
        del counts

        while len(lexicon) < 0.99*(len(lexicon) + len(unk)):
            for w in unk:
                unk.remove(w)
                lexicon.add(w)
                break
        del unk

        log('Replacing OOV words with <UNK>...')
        for l in lines_tokens:
            for i in xrange(len(l)):
                if l[i][0] != '<' and l[i] not in lexicon:
                    l[i] = '<UNK>'
        del lexicon

        log('Counting ngrams...')
        m_grams = dict()
        tokens = 0
        grams = [defaultdict(int) for _ in xrange(self.n+1)]
        voc = set()
        for i, words in enumerate(lines_tokens):
            for word in words:
                voc.add(word)
            num_words = len(words)
            tokens += num_words

            for l in xrange(1, self.n+1):
                for j in xrange (l, num_words+1):
                    a = tuple(words[j-l:j])
                    grams[l][a] += 1

        grams[0] = dict()
        grams[0][tuple()] = tokens
        self.set_voc(voc)

        log('Calculating probabilities...')
        for l in xrange(1, self.n+1):
            for gram, gram_count in grams[l].iteritems():
                log_prob = self._calculate_log_prob(l, grams, gram_count, grams[l-1][gram[:-1]])
                self.set_prob(l, gram, log_prob)

        # Calculate probabilities for unseen ngrams (after smoothing they get a nonzero value...)
        #seen_count = defaultdict(int)
        #for l in xrange(1, self.n):
        #    for gram in grams[l+1]:
        #        seen_count[gram[:-1]] += 1

        for l in xrange(2, self.n+1):
            for base_gram, base_gram_count in grams[l-1].iteritems():
        #        gram_seen_count = seen_count[base_gram]
        #        gram_other_count = self.voc_size - gram_seen_count
                log_prob = self._calculate_log_prob(l, grams, 0, base_gram_count)
                self.set_prob(l, base_gram + ('<OTHER>',), log_prob)

        # Pr(W_n | W_{n-1}) when C(W_n)=0, C(W_{n-1})=0
        self.prob_no_information = self._calculate_log_prob(2, grams, 0, 0)

        # Test
        #import random
        #base_word = random.choice(grams[3].keys())[:2]
        ##base_word = ('<s>', 'over')
        #print base_word
        #total = LOGZERO
        #c = 0
        #for gram in grams[3]:
        #    if gram[:2] == base_word:
        #        c += 1
        #        total = add_log(total, self.get_prob(gram))
        #        print gram, exp(self.get_prob(gram))
        #if self.lmbd:
        #    total = add_log(total, self.get_prob(base_word + ('<OTHER>',)) + log(self.voc_size - c))
        #print exp(total)

    def _calculate_log_prob(self, l, grams, gram_count, prev_gram_count):
        # No need to smooth 1-grams obviously...
        if l == 1 or self.smoothing == 'none' or (self.smoothing == 'ls' and self.lmbd == 0):
            if prev_gram_count == 0:
                # No prior information, assume uniform distribution
                log_prob = -log(self.voc_size)
            elif gram_count == 0:
                log_prob = LOGZERO
            else:
                log_prob = log(gram_count) - log(prev_gram_count)
        elif self.smoothing == 'ls':
            log_prob = log(gram_count + self.lmbd) - log(prev_gram_count + self.lmbd * self.voc_size)
        elif self.smoothing == 'wb':
            pass
        else:
            raise Exception('Invalid smoothing %s' % self.smoothing)

        return log_prob


def load_test_file(n, lines):
    n_grams = []
    for line in lines:
        try:
            l = ['<s>'] + tokenize(line) + ['</s>']
        except:
            continue
        n_grams.extend((l[max(0, x-n+1):x+1] for x in xrange(len(l))))
    return n_grams

def normalize(a):
    s = float(sum(a))
    for i in a:
        a[i] /= s
