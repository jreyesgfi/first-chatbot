import nltk
from nltk.stem.porter import PorterStemmer
#nltk.download('punkt') #pre-trained model

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass

if __name__ == '__main__':
    stemmer = PorterStemmer()
    a="How're you doing?"
    print(a)
    print(tokenize(a))
    words= ["organize","Organizes","organizing"]
    print([stem(word) for word in words])