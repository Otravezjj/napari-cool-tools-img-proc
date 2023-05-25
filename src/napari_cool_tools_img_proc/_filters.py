"""
This module contains code for filtering images
"""
from tqdm import tqdm
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch,viewer,device,memory_stats

def filter_bilateral(img:Image,kernel_size:int=1,s0:int=10,s1:int=10) -> Image:
    ''''''
    u_byte = img_as_ubyte(img)
    bilat = mean_bilateral(u_byte,disk(kernel_size),s0=s0,s1=s1)
    return bilat

def filter_median(img:Image,kernel_size:int=1,applications:int=1) -> Image:
    ''''''
    target = img
    for i in range(applications):
        img_med = median_filter(target,disk(kernel_size))
        target = img_med
    return img_med

def sharpen_um(img:Image, radius:float=1.0, amount:float=1.0, preserve_range=False, channel_axis=None) -> Image:
    ''''''
    sharp_img = unsharp_mask(img, radius=radius,amount=amount, preserve_range=preserve_range, channel_axis=channel_axis)
    return sharp_img

def filter_bilateral(img:Image,kernel_size:int=3,sc:float=0.1,s0:int=10,s1:int=10):
    """Implementation of bilateral filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        sc (float): sigma_color Standard deviation for grayvalue/color distance (radiometric similarity). A larger value results in averaging of pixels with larger radiometric differences
        s0 (int): standard deviation of fist dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        s1 (int): standard deviation of the 2nd dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        
    Returns:
        Image Layer that has been bilaterally filtered  with '_Bilat' suffix added to name.
    """
    filter_bilateral_thread(img=img,kernel_size=kernel_size,sc=sc,s0=s0,s1=s1)
    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def filter_bilateral_thread(img:Image,kernel_size:int=3,sc:float=0.1,s0:int=10,s1:int=10) -> Image:
    """Implementation of bilateral filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        sc (float): sigma_color Standard deviation for grayvalue/color distance (radiometric similarity). A larger value results in averaging of pixels with larger radiometric differences
        s0 (int): standard deviation of fist dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        s1 (int): standard deviation of the 2nd dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        
    Returns:
        Image Layer that has been bilaterally filtered  with '_Bilat' suffix added to name.
    """
    show_info(f'Bilateral Filter thread has started')
    output = filter_bilateral_pt_func(img=img,kernel_size=kernel_size,sc=sc,s0=s0,s1=s1)
    show_info(f'Bilateral Filter thread has completed')
    
    return output


def filter_bilateral_pt_func(img:Image,kernel_size:int=3,sc:float=0.1,s0:int=10,s1:int=10,border_type:str='reflect',color_distance_type:str='l1') -> Image:
    """Implementation of bilateral filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        sc (float): sigma_color Standard deviation for grayvalue/color distance (radiometric similarity). A larger value results in averaging of pixels with larger radiometric differences
        s0 (int): standard deviation of fist dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        s1 (int): standard deviation of the 2nd dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        
    Returns:
        Image Layer that has been bilaterally filtered  with '_Bilat' suffix added to name.
    """
    
    from kornia.filters import bilateral_blur
    
    name = img.name

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_Bilat"}

    # optional layer type argument
    layer_type = "image"

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:

        pt_data = torch.tensor(data,device=device)

        if data.ndim == 2:
            in_data = pt_data.unsqueeze(0).unsqueeze(0)
            blur_data = bilateral_blur(in_data,(kernel_size,kernel_size),sc,(s0,s1)).squeeze()
            out_data = blur_data.detach().cpu().numpy()
            layer = Layer.create(out_data,add_kwargs,layer_type)
        elif data.ndim == 3:
            for i in tqdm(range(len(pt_data)),desc="Bilateral Blur"):
                in_data = pt_data[i].unsqueeze(0).unsqueeze(0)
                pt_data[i] = bilateral_blur(in_data,(kernel_size,kernel_size),sc,(s0,s1)).squeeze()

            out_data = pt_data.detach().cpu().numpy()
            layer = Layer.create(out_data,add_kwargs,layer_type)

        return layer
        