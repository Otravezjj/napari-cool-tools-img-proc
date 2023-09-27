"""
This module contains code for denoising images
"""

from torchvision.transforms.functional import gaussian_blur
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.types import ImageData
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch,viewer,device,memory_stats
from napari_cool_tools_img_proc._normalization import normalize_data_in_range_pt_func
from tqdm import tqdm

def torchvision_diff_of_gaus_2d_data_func(data:ImageData, low_sigma:float=1.0, high_sigma:float=20.0, truncate=4.0):
    """Implementation of median filter function
    Args:
        img (Image): Image/Volume to be segmented.
        low_sigma (float): standard deviation for lower intensity gaussian filter
        high_sigma (float): standard deviation for higher intensity gaussian filter
        truncate (float): number of standard deviations to filter 
        
    Returns:
        Image Layer that has had difference of gaussians applied to it  with '_Band-pass' suffix added to name.
    """

    #Calculate kernel size to match Scipy ndimage module
    radius_low = round(truncate * low_sigma)
    radius_high = round(truncate * high_sigma)
    kernel_low = 2 * radius_low + 1
    kernel_high = 2 * radius_high + 1

    data_ten = torch.unsqueeze(torch.unsqueeze(torch.tensor(data,device=device),0),0)
    blur_low = gaussian_blur(data_ten,kernel_low)
    blur_high = gaussian_blur(data_ten,kernel_high)
    diff_gaus = blur_low - blur_high
    output = diff_gaus.detach().squeeze().cpu().numpy()
    norm_out = normalize_data_in_range_pt_func(output,0.0,1.0,numpy_out=True)

    return norm_out



def diff_of_gaus(img:Image, low_sigma:float=1.0, high_sigma:float=20.0, mode='nearest',cval=0, channel_axis=None, truncate=4.0, pt=False) -> Layer:
    """Implementation of median filter function
    Args:
        img (Image): Image/Volume to be segmented.
        low_sigma (float): standard deviation for lower intensity gaussian filter
        high_sigma (float): standard deviation for higher intensity gaussian filter
        mode (str): how input array is extended when filter overlaps border 
                    reflect, constant, nearest, mirror, wrap, grid-constant, grid-mirror, grid-wrap
                    for option descriptions refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
        cval (int): value to fill past edges in "constant" mode
        channel_axis (int or none): optional if None image assumed to be grayscale otherwise indicates axis that denotes color channels
        truncate (float): number of standard deviations to filter 
        pt (bool): flag indicatiing whether to use pytorch implementation
        
    Returns:
        Image Layer that has had difference of gaussians applied to it  with '_Band-pass' suffix added to name.
    """

    diff_of_gaus_thread(img=img,low_sigma=low_sigma,high_sigma=high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate,pt=pt)

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def diff_of_gaus_thread(img:Image, low_sigma, high_sigma=None, mode='nearest',cval=0, channel_axis=None, truncate=4.0, pt=False) -> Layer:
    """Implementation of median filter function
    Args:
        img (Image): Image/Volume to be segmented.
        low_sigma (float): standard deviation for lower intensity gaussian filter
        high_sigma (float): standard deviation for higher intensity gaussian filter
        mode (str): how input array is extended when filter overlaps border 
                    reflect, constant, nearest, mirror, wrap, grid-constant, grid-mirror, grid-wrap
                    for option descriptions refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
        cval (int): value to fill past edges in "constant" mode
        channel_axis (int or none): optional if None image assumed to be grayscale otherwise indicates axis that denotes color channels
        truncate (float): number of standard deviations to filter
        pt (bool): flag indicatiing whether to use pytorch implementation
        
    Returns:
        Image Layer that has had difference of gaussians applied to it  with '_Band-pass' suffix added to name.
    """
    show_info("Difference of Gaussian thread has started")
    output = diff_of_gaus_func(img=img,low_sigma=low_sigma,high_sigma=high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate,pt=pt)
    show_info("Difference of Gaussian thread has completed")
    return output

def diff_of_gaus_func(img:Image, low_sigma, high_sigma=None, mode='nearest', cval=0, channel_axis=None, truncate=4.0, pt=False) -> Layer:
    """Implementation of median filter function
    Args:
        img (Image): Image/Volume to be segmented.
        low_sigma (float): standard deviation for lower intensity gaussian filter
        high_sigma (float): standard deviation for higher intensity gaussian filter
        mode (str): how input array is extended when filter overlaps border 
                    reflect, constant, nearest, mirror, wrap, grid-constant, grid-mirror, grid-wrap
                    for option descriptions refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
        cval (int): value to fill past edges in "constant" mode
        channel_axis (int or none): optional if None image assumed to be grayscale otherwise indicates axis that denotes color channels
        truncate (float): number of standard deviations to filter
        pt (bool): flag indicatiing whether to use pytorch implementation
        
    Returns:
        Image Layer that has had difference of gaussians applied to it  with '_Band-pass' suffix added to name.
    """
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
            if pt:
                filtered_image = torchvision_diff_of_gaus_2d_data_func(data,low_sigma,high_sigma)
            else:
                dog_image = difference_of_gaussians(data,low_sigma,high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate)
                filtered_image = normalize_data_in_range_pt_func(dog_image,0.0,1.0,True)
            layer = Layer.create(filtered_image,add_kwargs,layer_type)
        elif data.ndim == 3:
            for i in tqdm(range(len(data)),desc="Band-pass(DoG)"):
                if pt:
                    data[i] = torchvision_diff_of_gaus_2d_data_func(data[i],low_sigma,high_sigma)
                else:
                    dog_image = difference_of_gaussians(data[i],low_sigma,high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate)
                    data[i] = normalize_data_in_range_pt_func(dog_image,0.0,1.0,True)

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