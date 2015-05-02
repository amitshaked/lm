#!/usr/bin/env python2.7

import argparse
import sys
import data
from collections import namedtuple



def calculate_perplexity(model, ngrams, mgrams):
	
	print len(ngrams), len(mgrams)
	print ngrams[2], mgrams[2]

	
	return 0

def main():
    
	parser = argparse.ArgumentParser(description="Build language model from corpus files")
	parser.add_argument('-i', '--input_file', type=str, required=True, help='The input corpus file')
	parser.add_argument('-m', '--language_model', type=str, required=True, help='The ouput model file')
	args = parser.parse_args()

	model = data.load_file(args.language_model)
	print "model: %d" % model.n
	ngrams, mgrams = data.load_file(args.input_file, model.n)

	per = calculate_perplexity(model, ngrams, mgrams)

	return 0

if __name__ == '__main__':
   
    sys.exit(main())