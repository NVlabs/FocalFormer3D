# Installation
## Prerequisite
1. `Python=3.8.12`
2. `Pytorch=1.9.1+cu111`
3. `CUDA=11.1`

### **Step. 1** Install MMLibraries.
```
pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu114/torch1.9.1/index.html # change torch/cuda versions
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
python setup.py install
```

### **Step. 2** Download and setup nuScenes and Waymo dataset following dataset preparation: [mmdet3d-nuScenes-dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes_det.html) and [mmdet3d-Waymo-dataset](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo_det.html). 

The following is a reference for setting up the Waymo dataset.
```
pip install waymo-open-dataset-tf-2-4-0

git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
cd waymo-od
git checkout bae19fa
# use the Bazel build system
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential
```

### **Step. 3** Clone FocalFormer3D. 
```
git clone https://github.com/NVlabs/FocalFormer3D.git focalformer3d
cd focalformer3d
```

