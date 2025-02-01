# Overveiw
The offical source code for paper - ``A Deep Learning Framework for Fetal Heart Tracking in Ultrasound Videos: Toward Enhanced CHD Detection''


## Environment Setup
Better to create virtual env to avoid environment variable issues on Windows.
```
# creat and activate conda virtual environment
conda create -n feht python=3.12
conda activate feht

# get repo and to the latest-stable branch
git clone 
git fetch --all
git checkout latest-stable
```
After that, please install pytorch first based on own conditions.
For example:
```
pip3 install torch torchvision torchaudio
```
Then install this tool as editable way
```
pip install -e .
```

## Dataset and Folder structures
The tool requires an index file for data sampling in training/inference. Example could be found in ``data/train-meta-example.pkl``. The structure of the index is basically a dictionary as follows:
```
{
    "train":[
        "suid": "scan unique id",
        "puid": "patient unique id",
        "annotation data": [...],  # bounding box data
        "data_path": [...],  # the RELATIVE path under data_root in the below mentioned config file.  
    ],
    "val": [...],  # same structure as above
    "test": [...],  # same structure as above
}
```
For training, a configuration file is needed, examples can be found in ``scripts/heartDetection/g6_112_L2.yaml``


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
feht-train -c ./scripts/heartDetection/g6_112_L2.yaml -g [GPU ID]
```
After start training, the log and checkpoints will be placed in ``logs`` folder

## Evaluation
Simply use the following command line for doing inference (by default, on the latest saved model):
```
feht-eval -p ./logs/g6_112_L2/ -g [GPU ID]
```
OR on the best saved model:
```
feht-eval -p ./logs/g6_112_L2/ -c best -g [GPU ID]
```
OR pick on of the saved model:
```
feht-eval -p ./logs/g6_112_L2/ -c [epoch number] -g [GPU ID]
```

## Pretrained Model 
If you would like to directly use our pretrained model for inference，please manually download from [HERE](https://drive.google.com/file/d/1cBBbQKAeQTZ5Iz9xUgajHMWZWGZXC91U/view?usp=sharing) and put it under ``feht/weights``

In addition, if you would like to integrate your own train model in to this tool, you can move your picked model from ``log/[your_experiment_name]/checkpoints/[your_select_model].pt`` to ``feht/weights`` as well. Better rename it to a new name for better recognition. Please check next section for more details

## Integrated Inference
The tool supports multiple ways for doing inference, depends on the usage of this tool:
* Inference via an index file (CSV): Please check the [CSV example](data/inference_index_example.csv) in ``data/inference_index_example.csv``. After the prediction, you can check [this example](feht/examples/how_to_use_the_bbox_prediction.ipynb) for how to use the predicted bounding boxes in your own dataloader.
```
feht-inference --csv data/inference_index_example.csv --model feht-l2-pretrained --output path/to/your/folder --gpu 0
```
* Integrate the inference in your code: Please check the example in [here](feht/examples/model_instance.ipynb).

## Software Update
This tool will be regularly maintained and updated. You can always using the following command line to update the software:
```
feht-update
```


