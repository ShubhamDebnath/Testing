file = open('data/text_corpus.txt', 'r')

text = file.read().lower
chars = sorted(list(set(text)))
print(len(chars))