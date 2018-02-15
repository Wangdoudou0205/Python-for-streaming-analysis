import nltk
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize

text='This is a test text'
sent_tokenize_list = sent_tokenize(text)
words = nltk.tokenize.word_tokenize(text)

word_tokenize('Hello World.')

from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))

a='I want to sleep!'
words=[]
for w in a:
    if w not in stopwords:
        words.append(w)

print(words)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
word2={'birds', 'hammers','women','men','celebrating'}
for w in word2:
    print(ps.stem(w))

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
document='I want to sleep right now'
tokenized = nltk.sent_tokenize(document)
print(tokenized)
PunktSentenceTokenizer(document).tokenize(document)
def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
        for subtree in chunked.subtrees():
            print(subtree)
 #chunked.draw()
    except Exception as e:
        print(str(e))

from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg
sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
for x in range(5):
    print(tok[x])

from nltk.corpus import wordnet
syns = wordnet.synsets(XXX)
print(syns[0].name())
print(syns[0].lemmas()[0].name())
syns = wordnet.synsets(YYY)
print(syns[0].lemmas()[0].name())
print(syns[0].lemmas()[0].name())
syns = wordnet.synsets(VVV)
print(syns[0].lemmas()[0].name())
print(syns[0].lemmas()[0].name())

synonyms = []
antonyms = []
for syn in wordnet.synsets(XXX):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(set(synonyms))
print(set(antonyms))

synonyms = []
antonyms = []

for syn in wordnet.synsets("bad"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))