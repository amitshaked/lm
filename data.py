#!/usr/bin/env python2.7

Nmodel = namedtuple('Nmodel', ['n', 'num_ngrams', 'ngrams'])
Ngram = namedtuple('Ngram', ['prob', 'words'])


class LanguageModel(object):
    def __init__(self, lines, size):
    	self.models = dict()
    	self.size = size
    	lines = lines[1:] # /data/

    	# read total ngrams num:
    	
    	for i in xrange(len(lines)):
    		if line.starts_with('\\'):
    			break
    		if line == '\n':
    			continue
    		a = lines[i].strip().split('=')
    		self.models.add(int(a[0][6]), Nmodel(int(a[0][6]), a[1]))    		

    	lines = lines[i+1:]
    	
    	#read ngrams
    	for i in xrange(self.models):
    		j=0    		
    		for j in xrange(gram.num_ngrams):
    			a = lines[j].split('\t')
    			self.models[i].ngrams.append(Ngram(a[0], a[1].split()))
    			
    		lines = lines[j+1:]

    def __getitem__(self, item):
        return self.models[item]

    def __len__(self):
        return len(self.models)

def load_corpus(input_file):
	pass

def build_model(corpus_file, smoothing='ls', lmbd=1):
	with open(input_file, 'rb') as f:
		probs = dict()
		lines = f.readlines()
		for line in lines:
			line = '<s> %s </s>' % line # append the start and end of the sentence
			words = line.split()
			num_words = len(words)
			for i in xrange (4, num_words):
				pass
