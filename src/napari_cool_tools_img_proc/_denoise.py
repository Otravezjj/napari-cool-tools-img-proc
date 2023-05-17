"""
This module is contains code for denoising images
"""

from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_img_proc import viewer

def diff_of_gaus(img:Image, low_sigma:float=1.0, high_sigma:float=25.0, mode='nearest',cval=0, channel_axis=None, truncate=4.0) -> Layer:
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
    from tqdm import tqdm

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 diminsions"
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
            for i in tqdm(range(len(data)),desc="Current image"):
                data[i] = difference_of_gaussians(data[i],low_sigma,high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate)

            layer = Layer.create(data,add_kwargs,layer_type)

        return layer