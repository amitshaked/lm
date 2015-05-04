#!/usr/bin/env python2.7
import argparse
import sys
import data
from collections import namedtuple


def calculate_prob_ls():
	return 0

def calculate_perplexity(model, ngrams):
	s, lmbd = model.get_smoothing()
	
	p = 1.0
	for i in xrange(len(ngrams)):
		if ngrams[i] in model[model.n]:
			prob = ngrams[i] in model[model.n][ngrams[i]]
		else:
			prob = 0
		if s == 'ls':
			prob = calculate_prob_ls()
		elif s == 'wb':
			pass
		else:
			raise
	return 0

def main():
	parser = argparse.ArgumentParser(description="Build language model from corpus files")
	parser.add_argument('-i', '--input-file', type=str, required=True, help='The input corpus file')
	parser.add_argument('-m', '--language-model', type=str, required=True, help='The ouput model file')
	args = parser.parse_args()

	model = data.LanguageModel.load(open(args.language_model))
	print "n-gram: %d" % model.n
	ngrams = data.load_test_file(model.n, open(args.input_file,'rb').readlines())

	per = calculate_perplexity(model, ngrams)

	return 0

if __name__ == '__main__':
    sys.exit(main())