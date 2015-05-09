#!/usr/bin/env python2.7
import re
import string
from collections import defaultdict, namedtuple
from functools import partial
from itertools import count
from progressbar import ProgressBar
from hashlib import sha1
from math import log10
import os
import sys

LOGZERO = -sys.maxint - 1
exp = partial(pow, 10)
log = lambda x: LOGZERO if x == 0 else log10(x)

ALPHABET = [chr(ord('a') + i) for i in xrange(ord('z') - ord('a') + 1)] + [chr(ord('A') + i) for i in xrange(ord('Z') - ord('A') + 1)]

EM_EPSILON = 10**-6
EPSILON = 10**-8

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
        self.interpolate = False
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

    def set_smoothing(self, s, lmbd=1.0):
        self.smoothing = s
        if self.smoothing == 'ls':
            self.lmbd = lmbd
        elif self.smoothing == 'wb':
            self.interpolate = True

    def get_smoothing(self):
        return self.smoothing, self.lmbd

    def get_prob(self, words):
        n = len(words)
        words = tuple(w if w[0] == '<' or w in self.voc else '<UNK>' for w in words)
        d = self.models[n]
        if words in d:
            return d[words]

        if len(words) > 1 and words[:-1] not in self.models[n-1]:
            # Previous words were not seen
            if self.prob_no_information != LOGZERO:
                # Use probability when there's no information
                return self.prob_no_information
            else:
                # Backoff
                return self.get_prob(words[1:])

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

    def _count_grams(self, lines_tokens):
        ''' Count ngrams in a given list of tokenized lines '''
        # T(w) in Witten-Bell smoothing
        # types_after[w] = set of all types that occur after w
        types_after = defaultdict(set)

        # N(w) in Witten-Bell smoothing
        # num_tokens_after[w] = number of tokens that occur after w
        num_tokens_after = defaultdict(int)

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
                    gram = tuple(words[j-l:j])
                    if l > 1:
                        # Account T(w) and N(w) in WB smoothing
                        prev = gram[:-1]
                        types_after[prev].add(gram[-1])
                        num_tokens_after[prev] += 1
                    grams[l][gram] += 1

        grams[0] = dict()
        grams[0][tuple()] = tokens
        return tokens, grams, voc, types_after, num_tokens_after

    def train_model(self, lines, silent=False):
        # Print to log if needed
        if silent:
            def info(s):
                pass
        else:
            def info(s):
                print s

        if self.interpolate:
            n_held_out = int(len(lines)*.1)
            held_out = lines[:n_held_out]
            data = lines[n_held_out:]
        else:
            data = lines

        info('Tokenizing...')
        data_tokens = [['<s>'] + tokenize(l) + ['</s>'] for l in data]
        if self.interpolate:
            held_out_tokens = [['<s>'] + tokenize(l) + ['</s>'] for l in held_out]

        # Build the lexicon as words that appear more than once, should be at least 99% of number of tokens
        info('Building lexicon...')
        lexicon = set()
        unk = set()
        counts = defaultdict(int)
        for l in data_tokens:
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

        info('Replacing OOV words with <UNK>...')
        for l in data_tokens:
            for i in xrange(len(l)):
                if l[i][0] != '<' and l[i] not in lexicon:
                    l[i] = '<UNK>'
        del lexicon

        info('Counting ngrams...')
        tokens, grams, voc, types_after, num_tokens_after = self._count_grams(data_tokens)
        self.set_voc(voc)

        num_types_after = defaultdict(int, ((x, len(types_after[x])) for x in types_after))
        num_types_after[tuple()] = self.voc_size
        num_tokens_after[tuple()] = tokens

        info('Calculating probabilities...')
        self.set_prob(1, ('<OTHER>',), LOGZERO)
        for l in xrange(1, self.n+1):
            for gram, gram_count in grams[l].iteritems():
                prev = gram[:-1]
                log_prob = self._calculate_log_prob(l, grams, gram_count, grams[l-1][prev],
                        num_types_after[prev], num_tokens_after[prev])
                self.set_prob(l, gram, log_prob)

        # Calculate probabilities for unseen ngrams (after smoothing they get a nonzero value...)
        for l in xrange(2, self.n+1):
            for base_gram, base_gram_count in grams[l-1].iteritems():
                log_prob = self._calculate_log_prob(l, grams, 0, base_gram_count, num_types_after[base_gram],
                        num_tokens_after[base_gram])
                self.set_prob(l, base_gram + ('<OTHER>',), log_prob)

        # Pr(W_n | W_{n-1}) when C(W_n)=0, C(W_{n-1})=0
        self.prob_no_information = self._calculate_log_prob(2, grams, 0, 0, 0, 0)

        if not self.interpolate:
            return

        info('Interpolating...')

        _, held_out_grams, _, held_out_types_after, held_out_num_tokens_after = \
                self._count_grams(held_out_tokens)

        for base_gram in grams[self.n-1]:
            # Use EM to calculate lambda-values which provide max log-likelihood on held-out data
            # Based on: https://www.cs.cmu.edu/~roni/11761/Presentations/degenerateEM.pdf
            if held_out_num_tokens_after[base_gram] == 0:
                # This base gram is so rare that it doesn't appear in the held-out data -- interpolation
                #   won't matter much in this case anyway!
                continue

            log_lmbds = [log(1) - log(self.n)]*self.n

            prev_loglikelihood = self._calculate_lmbds_loglikelihood(base_gram, log_lmbds, held_out_grams,
                    held_out_types_after, held_out_num_tokens_after)

            for t in count(1):
                # E-step
                # log_denoms[w] = lg(denominator for word w)
                log_denoms = dict()
                for w in held_out_types_after[base_gram]:
                    gram = base_gram + (w,)
                    log_denoms[w] = self._calculate_interpolated_prob(gram, log_lmbds)

                # M-step
                for j in xrange(self.n):
                    new_log_lmbd = LOGZERO
                    for w in held_out_types_after[base_gram]:
                        gram = base_gram + (w,)
                        val = log_lmbds[j] + self.get_prob(gram[-j-1:]) \
                                + log(held_out_grams[self.n][gram]) - log_denoms[w]
                        new_log_lmbd = add_log(new_log_lmbd, val)
                    log_lmbds[j] = new_log_lmbd - log(held_out_num_tokens_after[base_gram])

                # Check for convergence
                loglikelihood = self._calculate_lmbds_loglikelihood(base_gram, log_lmbds, held_out_grams,
                        held_out_types_after, held_out_num_tokens_after)
                assert loglikelihood >= prev_loglikelihood
                if loglikelihood - prev_loglikelihood <= EM_EPSILON:
                    break
                prev_loglikelihood = loglikelihood

            # Calculate the new interpolated probabilities
            total = LOGZERO
            for w in types_after[base_gram]:
                gram = base_gram + (w,)
                new_prob = self._calculate_interpolated_prob(gram, log_lmbds)
                self.set_prob(self.n, gram, new_prob)
                total = add_log(total, new_prob)

            # All other unseen probabilities - (1-Sum_w(Pr(w|base))) / Z(base)
            new_other_prob = log(1.0 - exp(total)) - log(self.voc_size - num_types_after[base_gram])
            self.set_prob(self.n, base_gram + ('<OTHER>',), new_other_prob)

        # Verify probabilities sum to 1
        info('Testing...')
        l = self.n
        for base_gram in grams[l-1].keys():
            if base_gram[-1] == '</s>':
                continue
            total = LOGZERO
            for w in types_after[base_gram]:
                gram = base_gram + (w,)
                total = add_log(total, self.get_prob(gram))
            total = add_log(total, self.get_prob(base_gram + ('<OTHER>',)) + \
                    log(self.voc_size - num_types_after[base_gram]))
            if abs(total-0.0) > EPSILON:
                raise Exception('Bad total for %s: %.32f' % (base_gram, exp(total)))


    def _calculate_log_prob(self, l, grams, gram_count, prev_gram_count, num_types_after, num_tokens_after):
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
            z = self.voc_size - num_types_after
            if gram_count == 0:
                if num_types_after == 0:
                    log_prob = LOGZERO
                else:
                    log_prob = log(num_types_after) - (log(z) + log(num_tokens_after + num_types_after))
            else:
                log_prob = log(gram_count) - log(num_tokens_after + num_types_after)
        else:
            raise Exception('Invalid smoothing %s' % self.smoothing)

        return log_prob

    def _calculate_interpolated_prob(self, gram, log_lmbds):
        return reduce(add_log, (log_lmbds[k] + self.get_prob(gram[-k-1:]) for k in xrange(self.n)))

    def _calculate_lmbds_loglikelihood(self, base, log_lmbds, grams, types_after, num_tokens_after):
        if num_tokens_after[base] == 0:
            return 0

        log_likelihood = 0
        for w in types_after[base]:
            gram = base + (w,)
            val = self._calculate_interpolated_prob(gram, log_lmbds)
            log_likelihood += grams[self.n][gram] * val
        log_likelihood /= num_tokens_after[base]
        return log_likelihood


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
