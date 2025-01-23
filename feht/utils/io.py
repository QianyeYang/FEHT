import torch, yaml
import pandas as pd
import numpy as np
from PIL import Image


def save_csv(data: pd.DataFrame, path: str) -> None:
    '''
    Save the data as csv file.
    '''
    data.to_csv(path, index=False)


def load_csv(path: str) -> pd.DataFrame:
    '''
    Load the data from csv file.
    '''
    return pd.read_csv(path)


def save_checkpoint(
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        epoch: int, 
        checkpoint_path: str
        ) -> None:
    '''
    Save the model checkpoint.
    '''
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, 
        checkpoint_path)


def load_checkpoint(
        checkpoint_path: str,
        map_location: torch.device | None = None
        ) -> dict:
    '''
    Load the model checkpoint.
    '''
    if map_location is None:
        return torch.load(checkpoint_path)
    else:
        return torch.load(checkpoint_path, map_location=map_location)


def save_yaml(
        data: dict, 
        path: str
        ) -> None:
    '''
    Save the data as yaml file.
    '''
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    

def load_yaml(
        path: str
        ) -> dict:
    '''
    Load the data from yaml file.
    '''
    with open(path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def save_gray_image(array: np.ndarray, path: str) -> None:
    '''
    Save the array as gray image.
    '''
    Image.fromarray(array).convert('L').save(path)


def save_rgb_image(array: np.ndarray, path: str) -> None:
    '''
    Save the array as rgb image.
    '''
    Image.fromarray(array).convert('RGB').save(path)