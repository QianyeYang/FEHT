import os, json
import pickle as pkl
import pandas as pd
import yaml


DIR_PROJECT:str = os.path.dirname(os.path.dirname(__file__))


def save_json(
        data: dict, 
        path: str, 
        overwrite: bool = False
        ) -> None:
    '''
    Save the data as json file.
    
    :param data: The data to be saved.
    :type data: dict
    :param path: The path of the json file.
    :type path: str
    :param overwrite: If True, overwrite the existing file.
    :type overwrite: bool
    '''
    if os.path.isfile(path) and not overwrite:
        raise FileExistsError(
            f"File {path} already exists! \
            Please set overwrite to True to overwrite the existing file."
            )
    
    with open(path, 'w') as file:
        json.dump(data, file)


def load_json(path: str) -> dict:
    '''
    Load the data from json file.
    
    :param path: The path of the json file.
    :type path: str

    :return: The data loaded from the json file.
    :rtype: dict
    '''
    with open(path, 'r') as file:
        data = json.load(file)

    return data


def save_pkl(
        data: object, 
        path: str, 
        overwrite: bool = False
        ) -> None:
    '''
    Save the data as pkl file.
    
    :param data: The data to be saved.
    :type data: object
    :param path: The path of the pkl file.
    :type path: str
    :param overwrite: If True, overwrite the existing file.
    :type overwrite: bool

    :return: None
    :rtype: None
    '''
    if os.path.isfile(path) and not overwrite:
        raise FileExistsError(
            f"File {path} already exists! \
            Please set overwrite to True to overwrite the existing file."
            )
    
    with open(path, 'wb') as file:
        pkl.dump(data, file)


def load_pkl(path: str) -> object:
    '''
    Load the data from pkl file.
    
    :param path: The path of the pkl file.
    :type path: str

    :return: The data loaded from the pkl file.
    :rtype: object
    '''
    with open(path, 'rb') as file:
        data = pkl.load(file)
    return data


def save_csv(
        data: pd.DataFrame, 
        path: str, 
        overwrite:bool = False
        ) -> None:
    '''
    Save the data as csv file.

    :param data: The data to be saved.
    :type data: pd.DataFrame
    :param path: The path of the csv file.
    :type path: str
    :param overwrite: If True, overwrite the existing file.
    :type overwrite: bool

    :return: None
    :rtype: None
    '''
    if os.path.isfile(path) and not overwrite:
        raise FileExistsError(
            f"File {path} already exists! \
            Please set overwrite to True to overwrite the existing file."
            )
    
    data.to_csv(path, index=False)


def load_csv(path: str) -> pd.DataFrame:
    '''
    Load the data from csv file.
    
    :param path: The path of the csv file.
    :type path: str

    :return: The data loaded from the csv file.
    :rtype: pd.DataFrame
    '''

    return pd.read_csv(path)


def save_yaml(
        data: dict, 
        path: str,
        overwrite: bool = False
        ) -> None:
    '''
    Save the data as yaml file.
    
    :param data: The data to be saved.
    :type data: dict
    :param path: The path of the yaml file.
    :type path: str
    :param overwrite: If True, overwrite the existing file.
    :type overwrite: bool

    :return: None
    :rtype: None
    '''
    if os.path.isfile(path) and not overwrite:
        raise FileExistsError(
            f"File {path} already exists! \
            Please set overwrite to True to overwrite the existing file."
            )
    
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    

def load_yaml(
        path: str
        ) -> dict:
    '''
    Load the data from yaml file.
    
    :param path: The path of the yaml file.
    :type path: str

    :return: The data loaded from the yaml file.
    :rtype: dict
    '''
    with open(path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    return data