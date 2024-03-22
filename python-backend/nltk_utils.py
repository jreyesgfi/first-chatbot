import nltk
from nltk.stem.porter import PorterStemmer
#nltk.download('punkt') #pre-trained model
import numpy as np

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

stemmer = PorterStemmer()
def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    bag = [1 if w in tokenized_sentence else 0 for w in all_words]
    return np.array(bag, dtype=np.float32)

if __name__ == '__main__':
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = bag_of_words(sentence, words)
    print(bag)