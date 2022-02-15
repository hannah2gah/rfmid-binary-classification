# -*- coding: utf-8 -*-
import numpy as np

#?#
def projection(img):
    '''
    Create vertical and horizontal projection of an image

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    vp : numpy array
    hp : numpy array

    '''

    vp = np.sum(img, axis=0)
    hp = np.sum(img, axis=1)
    return vp,hp
#?#

def crop_image(img_color, thres=(3000,3000)):
    '''
    Crop above the head and feet

    Parameters
    ----------
    img_color : RGB Image
    thres : int, int
        DESCRIPTION. The default is (3000,3000).

    Returns
    -------
    img_crop : RGB Image
        DESCRIPTION. Crop result
    row : array_like
    col : array_like

    '''

    img = img_color[:,:,0]
    h, w = img.shape
    mid1 = int(w/2)
    mid2 = int(h/2)

    if img.shape >= (2800,4200):
        thres = (30000,6000)
    else:
        thres = thres
 
    vp,hp = projection(img)
    check1_col = np.argwhere(vp[:mid1] < thres[0])
    check2_col = np.argwhere(vp[mid1:] < thres[0])
    check1_row = np.argwhere(hp[:mid2] < thres[1])
    check2_row = np.argwhere(hp[mid2:] < thres[1])  

    if check1_row.size == 0:
        row1 = 0
    else:
        row1 = int(check1_row[-1])
        
    if check2_row.size == 0:
        row2 = h-1
    else:
        row2 = int(mid2 + check2_row[0])
    
    if check1_col.size == 0:
        col1 = 0
    else:
        col1 = int(check1_col[-1])
    
    if check2_col.size == 0:
        col2 = w-1
    else:
        col2 = int(mid1 + check2_col[0])
        
    img_crop = img_color[row1:row2, col1:col2,:]
    row = (row1,row2)
    col = (col1,col2)
    return img_crop, row, col
#?#