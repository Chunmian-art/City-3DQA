import json

mode_path = "/workspace/UrbanQA/data/qa/sentence_mode/"
with open(f'{mode_path}/train.json', 'r') as file:
    data = json.load(file)
    print(len(data))