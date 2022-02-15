# -*- coding: utf-8 -*-
import pandas as pd

def assign_label(csv_path, image_path, sep=','):
    """
    Read csv and new column

    Parameters
    ----------
    csv_path : TYPE string
        DESCRIPTION.
    image_path : TYPE string
        DESCRIPTION.
    sep : TYPE, string, optional
        DESCRIPTION. The default is ','.

    Returns
    -------
    data_labels : TYPE
        DESCRIPTION.
    target_labels : TYPE
        DESCRIPTION.

    """
    df = pd.read_csv(csv_path,sep)
    data_labels = df.iloc[:,:2]
    target_labels = data_labels.iloc[:,-1]
    data_labels['image_path'] =  data_labels.apply(lambda row: (image_path + str(row['ID']) + '.png'), axis=1)
    return data_labels, target_labels

def 

