#!/usr/bin/env python2.7
from collections import defaultdict
import sys
import data
from eval import calculate_perplexity

def main():
    lambda_values = [ 0.0001, 0.001, 0.01 ] + [0.1*i for i in xrange(1,11)] + [ 2.0, 5.0, 10.0 ]
    languages = ['en', 'es', 'ca']
    n_values = [2,3,4]

    lang_lines = {lang: open(r'files\%s_text.corp' % lang, 'rb').readlines() for lang in languages}
    test_lines = {}
    for lang in languages:
        lines = open(r'files\%s.test' % lang, 'rb').readlines()
        for n in n_values:
            test_lines[(lang, n)] = data.load_test_file(n, lines)

    for n in n_values:
        print 'N: %d' % n
        print '=================='
        for lang in languages:
            print 'Language: %s' % lang
            print '=================='
            perplexities_per_test_lang = defaultdict(list)
            for lmbd in lambda_values:
                model = data.LanguageModel(n)
                model.set_smoothing('ls', lmbd)
                model.train_model(lang_lines[lang], silent=True)
                for test_lang in languages:
                    per = calculate_perplexity(model, test_lines[(test_lang, n)])
                    perplexities_per_test_lang[test_lang].append((lmbd, per))
            for test_lang in languages:
                print 'Test language: %s' % test_lang
                print '=================='
                for lmbd, per in perplexities_per_test_lang[test_lang]:
                    print '%.16f\t%.16f' % (lmbd, per)

    return 0

if '__main__' == __name__:
    sys.exit(main())
