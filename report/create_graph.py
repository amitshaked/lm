ls = []
while True:
    l = raw_input()
    if l == '':
        break
    ls.extend(l.split())
print "\n".join("(%s,%s)" % (ls[i],ls[i+1]) for i in xrange(0, len(ls),2))
