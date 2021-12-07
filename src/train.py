import json
from utils import tokenizer, stem, bag_or_words
import numpy as np
from CONST import BATCH_SIZE, NUM_WORKERS, HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS
from model import NeuralNet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


def main():
    with open('../data/intents.json', 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenizer(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '!', '.',',']
    all_words[:] = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
    tags[:] = sorted(set(tags))

    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_or_words(pattern_sentence, all_words)
        X_train.append(bag)

        y_train.append(tags.index(tag)) # don't have to care about one-hot encoding

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    # input_size = len(all_words)
    input_size = len(X_train[0])
    output_size = len(tags)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size = input_size, hidden_size=HIDDEN_SIZE, output_size=output_size).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for (train, label) in train_loader:
            x = train.to(device)
            y = train.to(device)

            out = model(x)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch + 1 % 100 == 0:
            print(f'The epoch {epoch + 1} with loss = {loss.item():.4f}')

    print(f'The final loss is {loss.item():.4f}')

    data = {
        "model_state":model.state_dict(),
        "input_size":input_size,
        "output_size":output_size,
        "hidden_size":HIDDEN_SIZE,
        "all_words":all_words,
        "tags":tags
    }
    file = "data.pth" # for pytorch
    torch.save(data, file)

    print(f'training complete')

if __name__ == "__main__":
    main()
