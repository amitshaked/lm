#!/usr/bin/env python2.7
from collections import defaultdict
import sys
import data
from eval import calculate_perplexity

SMOOTHING = 'wb'
NGRAM = 3

def main():
    if len(sys.argv) == 1:
        print 'Usage: %s result_file' % sys.argv[0]
        return 1
    languages = ['en', 'es', 'ca']
    models = []

    for lang in languages:
        print 'Training %s...' % lang
        model = data.LanguageModel(NGRAM)
        model.set_smoothing(SMOOTHING, 0.0001)
        model.train_model(open('files/%s_text.corp' % lang, 'rb').readlines(), silent=True)
        models.append(model)

    outf = open(sys.argv[1],'wb')

    for l in open('files/mixed.test','rb'):
        l = l.strip()
        l = data.load_test_file(NGRAM, (l,))

        smallest_per = None
        lang = None
        for i in xrange(len(models)):
            per = calculate_perplexity(models[i], l)
            if smallest_per is None or per < smallest_per:
                smallest_per = per
                lang = i
        outf.write('%d\n'%lang)
        outf.flush()

    outf.close()

    return 0

if '__main__' == __name__:
    sys.exit(main())
