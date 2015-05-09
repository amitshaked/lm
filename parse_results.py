#!/usr/bin/env python2.7
import sys

key = open('files/mixed.key','rb')
result = open(sys.argv[1],'rb')

languages = ['en', 'es', 'ca']

valid = [0,0,0]
total = [0,0,0]

for i in xrange(600):
    l1 = int(key.readline().strip())
    l2 = int(result.readline().strip())
    total[l1] += 1

    if l1 == l2:
        valid[l1] += 1
    elif l1==0:
        print l2

print 'en: %.10f' % (valid[0]/float(total[0]))
print 'es: %.10f' % (valid[1]/float(total[1]))
print 'ca: %.10f' % (valid[2]/float(total[2]))
