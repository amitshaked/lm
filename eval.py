#!/usr/bin/env python2.7
import argparse
import sys
import data
from collections import namedtuple
from math import exp


def calculate_perplexity(model, ngrams):
    lg_p = 0.0

    for ngram in ngrams:
        lg_p += model.get_prob(ngram)

    # lg(1/pow(prob, 1/N)) = lg(1) - (1/N)*lg(prob) = -(1/N)*lg(prob)
    if lg_p <= data.LOGZERO:
        raise Exception('Probability is zero, can\'t calculate perplexity!')
    return exp(-(1./len(ngrams)) * lg_p)

def main():
    parser = argparse.ArgumentParser(description="Build language model from corpus files")
    parser.add_argument('-i', '--input-file', type=str, required=True, help='The input corpus file')
    parser.add_argument('-m', '--language-model', type=str, required=True, help='The ouput model file')
    args = parser.parse_args()

    model = data.LanguageModel.load(open(args.language_model))
    ngrams = data.load_test_file(model.n, open(args.input_file,'rb').readlines())

    per = calculate_perplexity(model, ngrams)

    print '%.32f' % per

    return 0

if __name__ == '__main__':
    sys.exit(main())
