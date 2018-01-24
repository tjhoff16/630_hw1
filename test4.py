import re

s = "the big angry fox jumps over the large fence"
#
# print (re.split('[a-zA-Z]+', s))
#
# print (re.split('a', s))
#
# print (re.split ('\t', s))
#
# s= s.split()
# ss=''
# s.append ['the', 'big', 'hi', 'angry']
# ' '.join(s)
# print (ss)

z=[('2342324', ['the', '4', 'life'], '0'), ('234234', ['first', 'the', 'man'], '0')]

d ={}
x=0
for each_row in z:
    if each_row[2] == '0':
        for word in each_row[1]:
            d[word] = d.get(word, 0)+1

# print (d)
