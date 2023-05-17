"""
This module is contains code for adjusting image luminance
"""

from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_img_proc import viewer

def adjust_gamma(img:Image, gamma:float=1, gain:float=1) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gamma(float): Non negative real number.
        gain (float): Constant multiplier.
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name."""
    
    adjust_gamma_thread(img=img,gamma=gamma,gain=gain)
    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def adjust_gamma_thread(img:Image, gamma:float=1, gain:float=1) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gamma(float): Non negative real number.
        gain (float): Constant multiplier.
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name."""
    
    show_info(f"Adjust gamma thread started")
    output = adjust_gamma_func(img=img,gamma=gamma,gain=gain)
    show_info(f"Adjust gamma thread completed")
    return output

def adjust_gamma_func(img:Image, gamma:float=1, gain:float=1) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gamma(float): Non negative real number.
        gain (float): Constant multiplier.
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name."""
    
    from skimage.exposure import adjust_gamma

    data = img.data
    name = f"{img.name}_GC"
    layer_type = "image"
    add_kwargs = {"name": f"{name}"}

    gamma_corrected = adjust_gamma(data,gamma=gamma,gain=gain)

    layer = Layer.create(gamma_corrected,add_kwargs,layer_type)

    return layer
    

def adjust_log(img:Image, gain:float=1, inv:bool=False) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gain (float): Constant multiplier.
        inv (bool): If True performs inverse log correction instead of log correction.
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name."""
    
    adjust_log_thread(img=img,gain=gain,inv=inv)
    #return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def adjust_log_thread(img:Image, gain:float=1, inv:bool=False) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gain (float): Constant multiplier.
        inv (bool): If True performs inverse log correction instead of log correction.
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name."""
    
    show_info(f"Adjust log thread started")
    output = adjust_log_func(img=img,gain=gain,inv=inv)
    show_info(f"Adjust log thread completed")
    return output

def adjust_log_func(img:Image, gain:float=1, inv:bool=False) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gain (float): constant multiplier.
        inv (bool): If True performs inverse log correction instead of log correction.
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name."""
    
    from skimage.exposure import adjust_log
    from tqdm import tqdm

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 diminsions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:
        
        name = f"{img.name}_LC"
        layer_type = "image"
        add_kwargs = {"name": f"{name}"}

        if data.ndim == 2:
            log_corrected = adjust_log(data,gain=gain,inv=inv)
        elif data.ndim == 3:
            for i in tqdm(range(len(data)),desc="Current image"):
                data[i] = adjust_log(data[i],gain=gain,inv=inv)

        layer = Layer.create(data,add_kwargs,layer_type)

    return layer