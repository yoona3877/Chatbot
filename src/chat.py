import random
import json
import torch
from model import NeuralNet
from utils import bag_or_words, tokenizer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('../data/intents.json', 'r') as f:
        intents = json.load(f)

    file = "data.pth"
    data = torch.load(file)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]
    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)  # knows the learn parameters
    model.eval()

    bot_name = "Yoona"

    while True:
        sentence = input("Type: ")
        if sentence == "quit":
            break
        words = tokenizer(sentence)
        X = bag_or_words(words, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        first , pred = torch.max(output, dim=1)
        tag = tags[pred.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][pred.item()]
        if prob > 0.5:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    responses = intent['responses']
                    print(f"{random.choice(responses)}")
        else:
            print(f"I don't know")


if __name__ == "__main__":
    main()
