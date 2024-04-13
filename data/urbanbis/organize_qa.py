import json
import random
from collections import Counter

answer_dict = {
  "Allocate temporary residency options": ["Allocating temporary residency options", "Supplying Short-term Housing Solutions"],
  "Travel to distance": ["Traveling to distance"],
  "Acting as a polling station": ["Acting as a polling station", "Offering spaces for  public hearings"],
  "Areas for individual work": ["Affording areas for individual work", "Offering personal work zones"],
  "Providing retail space": ["Providing hospitality services", "Providing retail space"],
  "Purchasing tickets for travel": ["Purchasing tickets for travel", "Affording waiting transportation vehicles"],
  "Preserving heritage and traditions": ["Preserving heritage and traditions", "Displaying art and artifacts"],
  "Providing a means to connect two separate areas": ["Providing a means to connect two separate areas"],
  "Travel in the river": ["Traveling in the river"]
}


random.seed(114514)
urbanqa_v4_path = "/workspace/UrbanQA/data/urbanbis/urbanqa_v4.json"
urbanqa_v4 = json.load(open(urbanqa_v4_path, "r"))
random.shuffle(urbanqa_v4)
length = len(urbanqa_v4)
new_urbanqa_v4 = []
for item in urbanqa_v4:
    if isinstance(item["question"], str):
        new_urbanqa_v4.append(item)
del urbanqa_v4

train_mode = "example_mode"
train_set = []
test_set = []
val_set = []
all_length = len(new_urbanqa_v4)
for item in new_urbanqa_v4:
    if item["answer"] in answer_dict.keys():
        answer = answer_dict[item["answer"]]
    else:
        answer = [item["answer"]]
    city_name = item["area"].split("_")[0]
    save_item = {
        "template_type": item["template_type"],
        "answers": answer,
        "question": item["question"], 
        "question_id": item["id"], 
        "scene_id": item["area"],
        "hop": item["hop"]
    }
    train_set.append(save_item)
# for item in new_urbanqa_v4[1000: 1200]:
#     if item["answer"] in answer_dict.keys():
#         answer = answer_dict[item["answer"]]
#     else:
#         answer = [item["answer"]]
#     city_name = item["area"].split("_")[0]
#     save_item = {
#         "template_type": item["template_type"],
#         "answers": answer,
#         "question": item["question"], 
#         "question_id": item["id"], 
#         "scene_id": item["area"],
#         "hop": item["hop"]
#     }
#     val_set.append(save_item)
# for item in new_urbanqa_v4[1200:1400]:
#     if item["answer"] in answer_dict.keys():
#         answer = answer_dict[item["answer"]]
#     else:
#         answer = [item["answer"]]
#     city_name = item["area"].split("_")[0]
#     save_item = {
#         "template_type": item["template_type"],
#         "answers": answer,
#         "question": item["question"], 
#         "question_id": item["id"], 
#         "scene_id": item["area"],
#         "hop": item["hop"]
#     }
#     test_set.append(save_item)

# print(all_length, len(train_set), len(val_set), len(test_set))

with open(f"/workspace/UrbanQA/data/qa/{train_mode}/urbanqa_v5.json", "w") as f:
    json.dump(train_set, f)
# with open(f"/workspace/UrbanQA/data/qa/{train_mode}/val.json", "w") as f:
#     json.dump(val_set, f)
# with open(f"/workspace/UrbanQA/data/qa/{train_mode}/test.json", "w") as f:
#     json.dump(test_set, f)

# train_mode = "urban_mode"
# train_urban_mode = []
# test_urban_mode = []
# val_urban_mode = []
# # {'Longhua': 125602, 'Wuhu': 101921, 'Qingdao': 81944, 'Lihu': 78997, 'Yuehai': 60684, 'Yingrenshi': 316}
# for item in new_urbanqa_v4:
#     if item["answer"] in answer_dict.keys():
#         answer = answer_dict[item["answer"]]
#     else:
#         answer = [item["answer"]]
#     city_name = item["area"].split("_")[0]
#     save_item = {
#         "answers": answer,
#         "question": item["question"], 
#         "question_id": item["id"], 
#         "scene_id": item["area"],
#         "hop": item["hop"]
#     }
#     if city_name == "Lihu":
#         val_urban_mode.append(save_item)
#     elif city_name == "Yuehai":
#         test_urban_mode.append(save_item)
#     else:
#         train_urban_mode.append(save_item)
# print(len(train_urban_mode), len(val_urban_mode), len(test_urban_mode))


# with open(f"/workspace/UrbanQA/data/qa/{train_mode}/train.json", "w") as f:
#     json.dump(train_urban_mode, f)
# with open(f"/workspace/UrbanQA/data/qa/{train_mode}/val.json", "w") as f:
#     json.dump(val_urban_mode, f)
# with open(f"/workspace/UrbanQA/data/qa/{train_mode}/test.json", "w") as f:
#     json.dump(test_urban_mode, f)


# train_mode = "sentence_mode"
# train_set = []
# test_set = []
# val_set = []
# all_length = len(new_urbanqa_v4)
# for item in new_urbanqa_v4[:309783]:
#     if item["answer"] in answer_dict.keys():
#         answer = answer_dict[item["answer"]]
#     else:
#         answer = [item["answer"]]
#     city_name = item["area"].split("_")[0]
#     save_item = {
#         "answers": answer,
#         "question": item["question"], 
#         "question_id": item["id"], 
#         "scene_id": item["area"],
#         "hop": item["hop"]
#     }
#     train_set.append(save_item)
# for item in new_urbanqa_v4[309783: 309783+78994]:
#     city_name = item["area"].split("_")[0]
#     if item["answer"] in answer_dict.keys():
#         answer = answer_dict[item["answer"]]
#     else:
#         answer = [item["answer"]]
#     save_item = {
#         "answers": answer,
#         "question": item["question"], 
#         "question_id": item["id"], 
#         "scene_id": item["area"],
#         "hop": item["hop"]
#     }
#     val_set.append(save_item)
# for item in new_urbanqa_v4[309783+78994:]:
#     if item["answer"] in answer_dict.keys():
#         answer = answer_dict[item["answer"]]
#     else:
#         answer = [item["answer"]]
#     city_name = item["area"].split("_")[0]
#     save_item = {
#         "answers": answer,
#         "question": item["question"], 
#         "question_id": item["id"], 
#         "scene_id": item["area"],
#         "hop": item["hop"]
#     }
#     test_set.append(save_item)

# print(all_length, len(train_set), len(val_set), len(test_set))

# with open(f"/workspace/UrbanQA/data/qa/{train_mode}/train.json", "w") as f:
#     json.dump(train_set, f)
# with open(f"/workspace/UrbanQA/data/qa/{train_mode}/val.json", "w") as f:
#     json.dump(val_set, f)
# with open(f"/workspace/UrbanQA/data/qa/{train_mode}/test.json", "w") as f:
#     json.dump(test_set, f)