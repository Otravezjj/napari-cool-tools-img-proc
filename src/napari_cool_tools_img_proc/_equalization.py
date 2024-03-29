"""
This module contains code for equalizing image values
"""
import numpy as np
from tqdm import tqdm
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch,viewer,device,memory_stats
from napari_cool_tools_img_proc._normalization import normalize_in_range_pt_func

def clahe(img:Image, kernel_size=None,clip_limit:float=0.01,nbins=256,norm_min=0,norm_max=1,pt_K:bool=True) -> Layer:
    ''''''
    clahe_thread(img=img,kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins,norm_min=norm_min,norm_max=norm_max,pt_K=pt_K)

    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def clahe_thread(img:Image, kernel_size=None,clip_limit:float=0.01,nbins=256,norm_min=0,norm_max=1,pt_K:bool=True) -> Layer:
    ''''''
    show_info(f'Autocontrast (CLAHE) thread has started')
    if pt_K:
        output = clahe_pt_func(img=img,kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins,norm_min=norm_min,norm_max=norm_max)
        torch.cuda.empty_cache()
        memory_stats()
    else:
        output = clahe_func(img=img,kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins,norm_min=norm_min,norm_max=norm_max)
    show_info(f'Autocontrast (CLAHE) thread has completed')
    return output

def clahe_func(img:Image, kernel_size=None,clip_limit:float=0.01,nbins=256,norm_min=0,norm_max=1) -> Layer:
    ''''''
    from skimage.exposure import equalize_adapthist

    name = img.name

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_CLAHE"}

    # optional layer type argument
    layer_type = "image"

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:

        dtype_in = data.dtype
        norm_img = normalize_in_range_pt_func(img,norm_min,norm_max,in_place=False)
        norm_data = norm_img.data

        if data.ndim == 2:
            init_out = equalize_adapthist(norm_data,kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins)
            img_out = init_out.astype(dtype_in)
            layer = Layer.create(img_out,add_kwargs,layer_type)
        elif data.ndim == 3:
            for i in tqdm(range(len(data)),desc="CLAHE"):
                norm_data[i] = equalize_adapthist(norm_data[i],kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins)
            
            img_out = norm_data.astype(dtype_in)
            layer = Layer.create(img_out,add_kwargs,layer_type)

        #init_out = equalize_adapthist(norm_data,kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins)

        #img_out = init_out.astype(dtype_in)

        #layer = Layer.create(img_out,add_kwargs,layer_type)

        return layer
    
def clahe_pt_func(img:Image, kernel_size=None,clip_limit:float=40.0,nbins=256,norm_min=0,norm_max=1) -> Layer:
    """"""

    from kornia.enhance import equalize_clahe

    name = f"{img.name}_CLAHE"
    layer_type = "image"
    add_kwargs = {"name": f"{name}"}

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:

        dtype_in = data.dtype
        norm_img = normalize_in_range_pt_func(img,norm_min,norm_max,in_place=False)
        norm_data = norm_img.data
        pt_data = torch.tensor(norm_data,device=device)

        if data.ndim == 2:
            equalized = equalize_clahe(pt_data,clip_limit)
            out_data = equalized.detach().cpu().numpy()
            layer = Layer.create(out_data,add_kwargs,layer_type)
        elif data.ndim == 3:
            for i in tqdm(range(len(pt_data)),desc="CLAHE(PT)"):
                pt_data[i] = equalize_clahe(pt_data[i],clip_limit)
            
            out_data = pt_data.detach().cpu().numpy()
            layer = Layer.create(out_data,add_kwargs,layer_type)

        return layer
    
def match_histogram(target_histogram:Image,debug:bool=False):
    """"""
    from skimage.exposure import match_histograms

    target_data = target_histogram.data
    current_selection = list(viewer.layers.selection)
    
    for layer in current_selection:
        matched = match_histograms(layer.data,target_data,channel_axis=-1)
        layer.data[:] = matched[:]
    return layer