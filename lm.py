#!/usr/bin/env python2.7

import argparse
import sys
import data
from collections import namedtuple



def main():
    
	parser = argparse.ArgumentParser(description="Build language model from corpus files")
	parser.add_argument('-n', '--ngram', type=int, required=True, help='n-gram should be from 1 to 5')
	parser.add_argument('-i', '--input_file', type=str, required=True, help='The input corpus file')
	parser.add_argument('-o', '--output_file', type=str, required=True, help='The ouput model file')
	parser.add_argument('-s', '--smoothing', type=str, default='ls', help='The smoothing technique')
	parser.add_argument('-lmbd', '--lmbd', type=float, default=1.0, help='The lambda value for Lindstone\'s smoothing')
	args = parser.parse_args()

	lm = data.build_model(args.input_file, args.ngram, args.smoothing, args.lmbd)
	lm.dump(args.output_file)

	return 0

if __name__ == '__main__':   
    sys.exit(main())