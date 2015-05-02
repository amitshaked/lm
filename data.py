#!/usr/bin/env python2.7
import string
from collections import defaultdict
from collections import namedtuple
from progressbar import ProgressBar
from hashlib import sha1
import math
import os

import nltk
from nltk.util import ngrams
from nltk.util import flatten

Nmodel = namedtuple('Nmodel', ['num_ngrams', 'ngrams'])
Ngram = namedtuple('Ngram', ['prob', 'words'])

class LanguageModel(object):
	def __init__(self, n, lines):
		self.models = dict()
		
		lines = lines[1:] # /data/

		# read total ngrams num:
		for i in xrange(len(lines)):
			if lines[i].startswith('\\'):
				break
			if lines[i] == '\n':
				continue
			a = lines[i].strip().split('=')
			self.models[int(a[0][6])] = Nmodel(int(a[1]), [])
			self.n = int(a[0][6])    		

		lines = lines[i:]
		#read ngrams
		for j in xrange(len(self.models)):
			n = int(lines[0][1]) # \n-grams:
			for k in xrange(self.models[n].num_ngrams):
				a = lines[j+1].split('\t')
				self.models[n].ngrams.append(Ngram(a[0], a[1].split()))
				
			lines = lines[k:]

	def __getitem__(self, item):
		return self.models[item]

	def __len__(self):
		return len(self.models)

def load_test_file(n, lines):
	n_grams = []
	m_grams = []
	for line in lines:
		try:
			l = nltk.word_tokenize(line)
		except:
			continue
		n_grams.extend(list(ngrams(l, n, pad_right=True, \
			pad_left=True, pad_symbol='<s>'))[1:])
		m_grams.extend(list(ngrams(l, n-1, pad_right=True, \
			pad_left=True, pad_symbol='<s>')))
	return [n_grams, m_grams]


def calculate_prob_ls(n, n_grams, m_grams, gram, size, smoothing, lmbd):
	try:
		a = math.log(n_grams[gram] + lmbd*size, 10) - math.log(m_grams[gram[:-1]] + size, 10 )
	except:
		print gram, size, n_grams[gram], m_grams[gram[:-1]]
		return 0
	assert m_grams[gram[:-1]] >= n_grams[gram]
	return a

def build_model(corpus_file, output_file, n, smoothing='ls', lmbd=1):
	n_grams = dict()
	m_grams = dict()
	voc = []
	with open(corpus_file, 'rb') as f:
		lines = f.readlines()
		print 'Building model...'
		progress = ProgressBar(maxval=len(lines)).start()
		
		for i in xrange(len(lines)):			
			try:
				words = ['<s>'] + nltk.word_tokenize(lines[i]) + ['<s>']
			except:						
				continue				
			voc.extend(words)
			num_words = len(words)
			for j in xrange (n-1, num_words):			
				a = tuple(words[j-n+1:j+1])
				b = a[:-1]				
				n_grams.setdefault(a, 0)
				m_grams.setdefault(b, 0)
				n_grams[a] +=1
				m_grams[b] += 1
				
			progress.update(i)
		progress.finish()
		

	with open(output_file, 'wb') as f:
		size = len(set(voc))
		f.write('\\data\\\n')
		f.write('ngram %d=%d\n' % (n, size))
		f.write('\\%d-grams:\n' % n)
	
		for gram in n_grams.keys():			
			p = calculate_prob_ls(n, n_grams, m_grams, gram, size, smoothing, lmbd)
			f.write('%f\t%s\n' % (p ," ".join(gram)))


def load_file(path, n = 2):
	filename = os.path.split(path)[1]	
	if filename.endswith('lm'):
		loader, typename = LanguageModel, 'Language model'
	elif filename.endswith('test'):
		loader, typename = load_test_file, 'Test file'   
	else:
		raise NotImplementedError('Unknown file type')

	with open(path, 'rb') as f:
		lines = f.readlines()
		print 'Loading %s %s...' % (typename, filename)
		ret = loader(n, lines)
		return ret