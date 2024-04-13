# **Installation**

### 1.Download the code

### 2.Install environment using Docker

```bash
cd dockerfile
docker build -t cuda1101 .
docker run --gpus all --name city3dqa -v <code_path>:/workspace --ipc=host -it cuda1101 /bin/bash
```

### 3.Create the conda environment

```bash
conda create -n city3dqa python=3.8
```

### 4.Install Pytorch

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### 5.Install other necessary packages

```bash
pip install -r requirements.txt
```

If there is error in code, please try to adjust the transformers version to 4.29.2 or safetensors version to 0.3.0 to fix it.

### 6.Compile the CUDA modules for the PointNet++ backbone

```bash
cd lib/pointnet2
python setup.py install
```

# Dataset

### 1.Feature Data

Please download the feature data of **City-3DQA** from

Please download it and unzip it to data folder.

### 2.Point Cloud Data

The raw point cloud data is from [UrbanBIS](https://vcc.tech/UrbanBIS).

### 3.Scene Graph Feature Data

Please download the feature data of scene graph from

### 4.The project need to organize as follow:

- sg_cityu/
    - data/
        - qa/
            - sentence_mode/
            - urban_mode/
        - sg/
        - urbanbis/
            - urbanbis_data/
                - Lihu_Area1_aligned_bbox.npy
                - …
    - dockerfile/
    - models/
    - scene_graph/
        - Lihu_Area1/
            - edges.pth
            - …
    - Scripts

# Training

You can train **Sg-CityU** with the following code

```bash
bash train.sh
```
