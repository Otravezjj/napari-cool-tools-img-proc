"""
This module contains code for denoising images
"""

from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.types import ImageData
from napari.qt.threading import thread_worker
from napari_cool_tools_io import viewer
from tqdm import tqdm

def diff_of_gaus(img:Image, low_sigma:float=1.0, high_sigma:float=20.0, mode='nearest',cval=0, channel_axis=None, truncate=4.0) -> Layer:
    """"""
    
    diff_of_gaus_thread(img=img,low_sigma=low_sigma,high_sigma=high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate)

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def diff_of_gaus_thread(img:Image, low_sigma, high_sigma=None, mode='nearest',cval=0, channel_axis=None, truncate=4.0) -> Layer:
    """"""
    show_info("Difference of Gaussian thread has started")
    output = diff_of_gaus_func(img=img,low_sigma=low_sigma,high_sigma=high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate)
    show_info("Difference of Gaussian thread has completed")
    return output

def diff_of_gaus_func(img:Image, low_sigma, high_sigma=None, mode='nearest', cval=0, channel_axis=None, truncate=4.0) -> Layer:
    """"""
    from skimage.filters import difference_of_gaussians

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:
        name = f"{img.name}_Band-pass"
        add_kwargs = {"name":f"{name}"}
        layer_type = 'image'

        if data.ndim == 2:
            filtered_image = difference_of_gaussians(data,low_sigma,high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate)
            layer = Layer.create(filtered_image,add_kwargs,layer_type)
        elif data.ndim == 3:
            for i in tqdm(range(len(data)),desc="Band-pass(DoG)"):
                data[i] = difference_of_gaussians(data[i],low_sigma,high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate)

            layer = Layer.create(data,add_kwargs,layer_type)

        return layer
    
def denoise_tv(img:Image, weight:float=0.1) -> Layer:
    ''''''
    denoise_tv_thread(img=img,weight=weight)
    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def denoise_tv_thread(img:Image, weight:float=0.1) -> Layer:
    ''''''
    show_info(f'Denoise Total Variation thread has started')
    denoise_data = denoise_tv_func(data=img.data,weight=weight)
    print("\n\nWe MADE IT HERE!!\n\n")
    name = f"{img.name}_TV"
    add_kwargs = {"name":f"{name}"}
    layer_type = 'image'
    layer = Layer.create(denoise_data,add_kwargs,layer_type)
    show_info(f'Denoise Total Variation thread has completed')
    return layer

def denoise_tv_func(data:ImageData, weight:float=0.1): #-> ImageData:
    """"""
    from skimage.restoration import denoise_tv_chambolle

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:
        tvd = data.copy()

        if data.ndim == 2:
            tvd = denoise_tv_chambolle(tvd, weight=weight,eps =0.0002)
        elif data.ndim == 3:
            for i in tqdm(range(len(data)),desc="Denoise(TV)"):
                tvd[i] = denoise_tv_chambolle(tvd[i], weight=weight,eps =0.0002)

        return tvd