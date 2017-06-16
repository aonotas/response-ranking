import sys

filename1 = sys.argv[1]
filename2 = sys.argv[2]


def build_vocab(filename):
    vocab = []
    for l in open(filename):
        word = l.decode('utf-8').strip()
        vocab.append(word)
    vocab = set(vocab)
    return vocab

vocab1 = build_vocab(filename1)
vocab2 = build_vocab(filename2)


common_words = vocab1.intersection(vocab2)

vocab1_only = vocab1.difference(common_words)
vocab2_only = vocab2.difference(common_words)

vocab_all = vocab1.union(vocab2)


print '---------------------'
print 'vocab1:', len(vocab1)
print 'vocab2:', len(vocab2)
print '---------------------'
print 'vocab1_only:', len(vocab1_only)
print 'vocab2_only:', len(vocab2_only)
print '---------------------'
print 'common_words:', len(common_words)
print 'vocab_all:', len(vocab_all)
print '---------------------'


print '---------------------'
print ' Examples'
print '---------------------'
print 'common_words:', len(common_words)
for w in list(common_words)[:10]:
    print ' ' + w
