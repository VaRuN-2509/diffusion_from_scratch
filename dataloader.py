import pandas as pd
import numpy as np

def dino():
    data = pd.read_csv("Datashape.tsv",sep='\t',header=0)
    data = data[data['dataset']=='dino']
    data = np.array(data)
    data = data[:,1:].astype(np.float32)

    return data

def away():
    data = pd.read_csv("Datashape.tsv",sep='\t',header=0)
    data = data[data['dataset']=='away']
    data = np.array(data)
    data = data[:,1:].astype(np.float32)

    return data

def circle():
    data = pd.read_csv("Datashape.tsv",sep='\t',header=0)
    data = data[data['dataset']=='circle']
    data = np.array(data)
    data = data[:,1:].astype(np.float32)

    return data

def x_shape():
    data = pd.read_csv("Datashape.tsv",sep='\t',header=0)
    data = data[data['dataset']=='x_shape']
    data = np.array(data)
    data = data[:,1:].astype(np.float32)

    return data
def bullseye():
    data = pd.read_csv("Datashape.tsv",sep='\t',header=0)
    data = data[data['dataset']=='bullseye']
    data = np.array(data)
    data = data[:,1:].astype(np.float32)

    return data

def dots():
    data = pd.read_csv("Datashape.tsv",sep='\t',header=0)
    data = data[data['dataset']=='dots']
    data = np.array(data)
    data = data[:,1:].astype(np.float32)

    return data

def h_lines():
    data = pd.read_csv("Datashape.tsv",sep='\t',header=0)
    data = data[data['dataset']=='h_lines']
    data = np.array(data)
    data = data[:,1:].astype(np.float32)

    return data

def star():
    data = pd.read_csv("Datashape.tsv",sep='\t',header=0)
    data = data[data['dataset']=='star']
    data = np.array(data)
    data = data[:,1:].astype(np.float32)

    return data

def get_dataset(dataset_name):
    """Get dataset based on name from config"""

    if dataset_name == "dino":
        return dino()
    elif dataset_name == "h_lines":
        return h_lines()
    elif dataset_name == "circle":
        return circle()
    elif dataset_name == "bullseye":
        return bullseye()
    elif dataset_name == "dots":
        return dots()
    elif dataset_name == "away":
        return away()
    elif dataset_name == "x_shape":
        return x_shape()
    elif dataset_name == "star":
        return star()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

