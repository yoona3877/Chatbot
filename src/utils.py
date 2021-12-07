import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer = PorterStemmer()

def tokenizer(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_or_words(tokenized_sentence, all_words):
    """
    :param tokenized_sentence:
    :param all_words:
    :return:
    """
    tokenenized_sentences = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenenized_sentences:
            bag[idx] = 1.0

    return bag

