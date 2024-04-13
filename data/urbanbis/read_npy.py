import numpy as np
import json
"""
data_path = "/workspace/ScanQA/data/scannet/meta_data/scannet_means.npz" 
# arr_0 (18, 3)
mesh_vertices = np.load("/workspace/ScanQA/data/scannet/scannet_data/scene0024_00_aligned_vert.npy") # axis-aligned
# for key in mesh_vertices:
#     print(key.shape) # 9
# print(len(mesh_vertices)) # 50000
instance_labels = np.load("/workspace/ScanQA/data/scannet/scannet_data/scene0024_00_ins_label.npy")
# for key in instance_labels:
#     print(key) # label
# print(len(instance_labels)) # 50000
semantic_labels = np.load("/workspace/ScanQA/data/scannet/scannet_data/scene0024_00_sem_label.npy")
# for key in semantic_labels:
#     print(key) # label
# print(len(semantic_labels)) # 50000
instance_bboxes = np.load("/workspace/ScanQA/data/scannet/scannet_data/scene0024_00_aligned_bbox.npy")
for key in instance_bboxes:
    print(key.shape) # 8 
print(len(instance_bboxes)) # 49
"""

# path = "/workspace/UrbanQA/data/urbanbis/meta_data/scannet_means.npz"
# scannet_means = np.load(path)
# for key in scannet_means:
#     print(key)
#     print(scannet_means[key].shape)
path = "/workspace/UrbanQA/data/qa/urban_mode/train.json"
data = json.load(open(path, "r"))
new_data = []
i = 0
for item in data:
    if not isinstance(item["question"], str):
        print(item)
        print(item["question"])
        print(type(item["question"]))
        i += 1
    else:
        new_data.append(item)
with open("/workspace/UrbanQA/data/qa/urban_mode/train.json", "w") as f:
    json.dump(new_data, f)
print(i)