import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from model import NeuralNet

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
        Y_train.append(label)

    X_train = np.array(X_train,dtype=np.float32)
    Y_train = np.array(Y_train,dtype=np.int64) #long
    return X_train, Y_train


class ChatDataset(Dataset):
    def __init__(self,xy,tags):
        X_train, Y_train = create_training_sets(xy,tags)
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


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
    ## hyperparameters
    batch_size = 8
    num_workers = 0
    hidden_size = 8
    output_size = len(tags)
    input_size = len(all_words)
    learning_rate = 0.001
    num_epochs = 1000

    ## datasets
    dataset = ChatDataset(xy, tags)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(input_size, len(dataset.x_data[0]))
    print(output_size, tags)

    ## model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    ## loss and optimized
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        for (words,labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            #backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1)%100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')

    print(f'final loss, loss={loss.item():.4f}')