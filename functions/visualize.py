# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model

#%% Filter
def get_layers(n, model):
    """
    Get layers weights of CNN architecture, then normalize it

    Parameters
    ----------
    n : TYPE int.
        DESCRIPTION. index layer
    model : TYPE 
        DESCRIPTION. CNN architecture

    Returns
    -------
    w_norm : TYPE float
        DESCRIPTION. normalize weight

    """
    layer = model.layers[n]
    w , b = layer.get_weights()
    print('Layer name: ', layer.name, ' Weight shape: ', w.shape)
    
    # normalize filter values to 0-1 
    w_min, w_max = w.min() , w.max()
    w_norm = (w - w_min) / (w_max - w_min)
    
    return w_norm

def plot_filters(w_norm, num_fil=4):
    '''
    Plot each filter in convolutional layers

    Parameters
    ----------
    w_norm : TYPE, float
        DESCRIPTION. normalise weight
    num_fil : TYPE, int, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    ax : TYPE plt
        DESCRIPTION.

    '''
    idx_plot = 1
    plt.figure(figsize=(6,2*num_fil))
    
    for i in range(num_fil):
        f = w_norm[:,:,:,i]
        
        # pyplot each channel separately
        for j in range(3):
            ax = plt.subplot(num_fil, 3, idx_plot)
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.imshow(f[:,:,j], cmap='viridis')
            idx_plot += 1
            
    plt.show()
    
    return ax

#%% Feature Map
def get_features(inputs, outputs, img):
    '''
    Get feature map on convolutional layer

    Parameters
    ----------
    inputs : TYPE 
        DESCRIPTION.
    outputs : TYPE
        DESCRIPTION.
    img : TYPE uint8
        DESCRIPTION.

    Returns
    -------
    model_layer : TYPE model
        DESCRIPTION.
    feat_maps : TYPE ndarray
        DESCRIPTION.

    '''
    model_layer = Model(inputs, outputs)
    
    feat_maps = model_layer.predict(img)
    
    return  model_layer,feat_maps

def plot_features(feat_maps,n_row=4, n_col=4):
    '''
    Plot each feature maps

    Parameters
    ----------
    feat_maps : TYPE array
        DESCRIPTION.
    n_row : TYPE, optional
        DESCRIPTION. The default is 4.
    n_col : TYPE, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    ax : TYPE pyplot
        DESCRIPTION.

    '''
    idx_plot = 1
    
    plt.figure(figsize=(3*n_col, 3*n_row))
    for _ in range(n_row):
        for _ in range(n_col):
            ax = plt.subplot(n_row, n_col, idx_plot)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # plot filter channel in grayscale
            plt.imshow(feat_maps[0,:,:, idx_plot-1], cmap='gray')
            idx_plot += 1
    
    plt.show()
    
    return ax