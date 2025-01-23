# Overveiw
The offical source code for paper - ``A Deep Learning Framework for Fetal Heart Tracking in Ultrasound Videos: Toward Enhanced CHD Detection''


## Environment Setup
Better to create virtual env to avoid environment variable issues on Windows.
```
# creat and activate conda virtual environment
conda create -n feht python=3.12
conda activate feht

# get repo and to the latest-dev branch
git clone 
git fetch --all
git checkout latest-dev
```
After that, please install pytorch first based on own conditions.
For example:
```
pip3 install torch torchvision torchaudio
```


# install as editable way
pip install -e .
```

## Dataset and Folder structures


## Annotation Tool
Please visit https://anonymous.4open.science/r/abcdC3D5 to access our customized US video annotation tool.

## Training
After data is prepared, the following command can be used for training a heart tracking model:
```
cd /path/to/this/repo/on/your/local/machine/
feht-train -c /path/to/the/configuration/file
```
For example:
```
feht-train -c ./scripts/heartDetection/g6_112_L2.yaml
```
After start training, 


## Pretrained Model
If you would like to directly use our pretrained model for inference，please manually download from here and put it in ``here``

## Inference

