import operator

p = {}

p[0]=4
p[2]=2
p[3]=3
p[4]=6

sort_class_count = sorted(p.items(), key=operator.itemgetter(1), reverse=True)
q = sort_class_count[0]
qq=q[0]

