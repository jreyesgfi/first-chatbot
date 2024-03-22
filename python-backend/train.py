import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

def tagging_words(intents,tags,all_words,xy, ignore_words=['.',',']):
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            words = tokenize(pattern)
            words = [stem(word) for word in words if word not in ignore_words]
            all_words.extend(words)
            xy.append((words, tag))
    return

def create_training_sets(xy,tags):
    X_train = []
    Y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return X_train, Y_train

if __name__ == '__main__':

    #opening files
    with open('intents.json','r') as f:
        intents = json.load(f)

    # tagging words
    tags = []
    all_words = []
    xy = []
    ignore_words = ['?','!','.',',']
    tagging_words(intents,tags,all_words,xy,ignore_words)
    

    # stemming and excluding duplicates
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # training
    X_train,Y_train = create_training_sets(xy,tags)
