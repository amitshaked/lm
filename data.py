#!/usr/bin/env python2.7
import string
from collections import defaultdict
from collections import namedtuple
from progressbar import ProgressBar
from hashlib import sha1
import math
import os
import utils
import nltk
from nltk.util import ngrams
from nltk.util import flatten

class LanguageModel(object):

	def __init__(self, n, lines = None, voc = None, smoothing = None, lmbd = None):		
		self.smoothing = smoothing
		self.lmbd = lmbd
		self.models = dict()
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

	def set_prob(self, n, words, prob):
		w = tuple(words)
		if n not in self.models:
			self.models[n] = {}		
		self.models[n][w] = prob #math.log(prob, 10)

	def set_smoothing(self, s, lmbd):
		self.smoothing = s
		self.lmbd = lmbd

	def get_smoothing(self, s, lmbd):
		return self.smoothing, self.lmbd

	def get_prob(self, n, words):
		return self.models[n][tuple(words)]

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
			for n in self.models:
				f.write('\\%d-grams:\n' % n)	
				for words in self.models[n]:			
					f.write('%.10f\t%s\n' % (self.models[n][words] ," ".join(words)))

def load_test_file(n, lines):
	n_grams = []
	for line in lines:
		try:
			l = ['s'] + nltk.word_tokenize(line) + ['/s']
		except:
			continue
		n_grams.extend(list(ngrams(l, n)))
	return n_grams


def calculate_prob_ls(n, n_grams, m_grams, gram, size, smoothing, lmbd):
	try:
		a = math.log(n_grams[gram] + lmbd*size, 10) - math.log(m_grams[gram[:-1]] + size, 10 )
	except:
		print gram, size, n_grams[gram], m_grams[gram[:-1]]
		return 0
	assert m_grams[gram[:-1]] >= n_grams[gram]
	return a

def build_model(n, lines):
	m_grams = dict()
	voc = []
	lm = LanguageModel(n)
	
	print 'Building model...'
		
	progress = ProgressBar(maxval=len(lines)).start()			
	grams = dict()
	voc=[]
	for i in xrange(len(lines)):			
		try:
			words = ['<s>'] + nltk.word_tokenize(lines[i]) + ['</s>']
		except:						
			continue				
		voc.extend(words)
		num_words = len(words)
		
		for l in xrange(1, n+1):
			if l not in grams:
				grams[l] = dict()
			for j in xrange (l, num_words+1):			
				a = tuple(words[j-l:j])
				grams[l].setdefault(a, 0)
				grams[l][a] +=1				
		progress.update(i)		
	progress.finish()
			
	grams[0] = dict()
	grams[0][tuple([])] = len(voc)
	voc = list(set(voc))
	lm.set_voc(voc)
	
	for l in xrange(1, n+1):
		for gram in grams[l]:
			p = float(grams[l][gram]/grams[l-1][gram[:-1]])
			lm.set_prob(l, gram, p)
		#normalize(lm[l])

	return lm

def load_file(path, n = 2):
	filename = os.path.split(path)[1]	
	if filename.endswith('lm'):
		loader, typename = LanguageModel, 'Language model'
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
    s = 0.0
    for i in a:
        s += a[i]
    for i in a:
        a[i] /= s