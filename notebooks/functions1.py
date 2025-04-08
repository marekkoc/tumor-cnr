"""
TUMOR-CNR

Module with usueful functions.

C: 2021.11.09 / U:2021.11.09
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def im_info(im, name='image'):
    """
    The basic info about image.
    
    Parameters:
    ---------------
    im - image,
    name - displayed image name (hepful in multiple use in the row)
    skip_zeros - skip voxels with zero values
    
    C: 2021.11.06 / U:2021.11.30
    """
    print(f'*** {name.upper()} ***,\tmax={im.max()}, min={im.min()}, mean={im.mean():.2f}, shape={im.shape}, #voxels={im.size}')


def create_rgb(img, roi=None, dil=None, slice=14, verbose=True):
    """
    Create RGB image from selected slices of `img`, `roi` and `dilated roi` images.
    
    Each slice creates single color channel in RGB image:
        R -> dilated mask,
        G -> roi mask,
        B -> image.
    
    C: 2021.11.07 / U: 2021.11.07
    """
    i = img[:,:,slice]
    # rescale to 0-1 if brightness > 1
    if i.max() > 1:
        i = (i - i.min()) / (i.max() - i.min()).astype('float32')
        if verbose: print('Rescaled "img"')
    
    # roi image
    if  isinstance(roi, np.ndarray):
        r = roi[:,:,slice]
        # rescale to 0-1 if brightness > 1
        if r.max() > 1:
            r = (r - r.min()) / (r.max() - r.min()).astype('float32')
            if verbose: print('Rescaled "roi"')
            
        # convert to boolean values, we will use these matrices as a matrix indices        
        r = np.asarray(r, dtype='bool')
        r_mask = i.copy()
        r_mask[r] = 1
    else:
        r_mask = i.copy()
        
    # dil image
    if isinstance(dil, np.ndarray):
        d = dil[:,:,slice]
        # rescale to 0-1 if brightness > 1
        if d.max() > 1:
            d = (d - i.min()) / (d.max() - d.min()).astype('float32');
            if verbose: print('Rescaled "dil"')
        # convert to boolean values, we will use these matrices as a matrix indices 
        d = np.asarray(d, dtype='bool')
        d_mask = i.copy()
        d_mask[d] = 1   
    else:
        d_mask = i.copy()
    
    # prepare 2D RGB image (for selected slice)
    rows, cols, slices = img.shape
    im3 = np.zeros((rows,cols,3), dtype=img.dtype)    
    
    # assemble RGB
    im3[:,:,0] = d_mask
    im3[:,:,1] = r_mask
    im3[:,:,2] = i
    return im3


def show_images_from_list(ax, img, roi, masks_lst, names_lst, slice_nr, title, fontsize=22):
    """
    Display images from an image list.
    
    PARAMETERS:
    ---------------------------------
    ax - matrix of axes,
    img - 3d image,
    roi - 3d roi mask,
    masks - list of mask images (dils, balls,...),
    names - names of mask images,
    slice_nr - slice nr to display,
    title - title of the whole plot.
    
    
    
    C: 2021.11.09 / U: 2021.11.09
    """
    axf = ax.flat[:]
    for k in range(len(masks_lst)):
        rgb = create_rgb(img, roi, masks_lst[k], verbose=False, slice=slice_nr)
        axf[k].imshow(rgb)
        axf[k].set_title(f'{names_lst[k]} (sl={slice_nr})', fontsize=fontsize-2)
        plt.suptitle(title, fontsize=fontsize, fontweight='bold')

    for k in range(len(masks_lst), len(axf)):
        axf[k].set_axis_off()
        
        
def show_images_from_list2(ax, img, roi, masks_lst, names_lst, slice_nr, title, fontsize=22):
    """
    Display images from an image list.
    
    PARAMETERS:
    ---------------------------------
    ax - matrix of axes,
    img - 3d image,
    roi - 3d roi mask,
    masks - list of mask images (dils, balls,...),
    names - names of mask images,
    slice_nr - slice nr to display,
    title - title of the whole plot.
    
    
    
    C: 2021.11.09 / U: 2021.11.09
    """
    axf = ax.flat[:]
    for k in range(len(masks_lst)):
        #rgb = create_rgb(img, roi, masks_lst[k], verbose=False, slice=slice_nr)
        
        i2 = img[:,:,slice_nr]
        r2 = roi[:,:,slice_nr]
        m2 = masks_lst[k][:,:,slice_nr]
        
        if i2.max() > 1:
            i2 = (i2 - i2.min()) / (i2.max() - i2.min()).astype('float32')
        
        rows, cols, slices = img.shape
        im3 = np.zeros((rows,cols,3), dtype=img.dtype)   
        im3[:,:,0] = i2
        im3[:,:,1] = i2
        im3[:,:,2] = i2
        
        rim = m2-r2        
        #roi - cyian #00FFFF
        im3[:,:,1] = np.where(r2==1, 1, im3[:,:,1])
        im3[:,:,2] = np.where(r2==1, 1, im3[:,:,2])
        
        #rim - magenta #FF00FF
        im3[:,:,0] = np.where(rim==1, 1, im3[:,:,0])
        im3[:,:,2] = np.where(rim==1, 1, im3[:,:,2])
        
        axf[k].imshow(im3, cmap='gray')
        axf[k].set_title(f'{names_lst[k]} (sl={slice_nr})', fontsize=fontsize-2)
        plt.suptitle(title, fontsize=fontsize, fontweight='bold')

    for k in range(len(masks_lst), len(axf)):
        axf[k].set_axis_off()
        

def crop_image_inside_bigger_mask(img, roi, mask):
    """
    Crop a part of the `img` and `roi` images that are under `mask` (usulally dilated roi).
    
    Parameters:
    -----------------------
    img - 3D image,
    roi - 3D roi image,
    mask - 3D binary mask - usually dilated roi.
    
    
    Return:
    ---------------------------
    cropped images:
        img_c, roi_c, mask_c
    
    C: 2021.11.08 / U: 2021.11.08
    """
    
    xy = mask.max(2)
    #print(xy.shape)
    xy_idx = np.where(xy==1)
    y1max,y1min = xy_idx[0].max(), xy_idx[0].min()
    x1max,x1min = xy_idx[1].max(), xy_idx[1].min()

    xz = mask.max(1)
    #print(xz.shape)
    xz_idx = np.where(xz==1)
    x2max,x2min = xz_idx[0].max(), xz_idx[0].min()
    z2max,z2min = xz_idx[1].max(), xz_idx[1].min()

    # crop images 
    img_c = img[y1min:y1max, x1min:x1max, z2min:z2max]
    roi_c = roi[y1min:y1max, x1min:x1max, z2min:z2max]
    mask_c = mask[y1min:y1max, x1min:x1max, z2min:z2max]
    
    return img_c, roi_c, mask_c


def get_image_under_single_mask(img, mask, crop=False):
    """
    Get image from inside of a single mask. If crop=True, return cropped image to the mask dimensions.
    
    C: 2021.11.09 / U: 2021.11.09
    """
    
    if crop:
        xy = mask.max(2)
        #print(xy.shape)
        xy_idx = np.where(xy==1)
        y1max,y1min = xy_idx[0].max(), xy_idx[0].min()
        x1max,x1min = xy_idx[1].max(), xy_idx[1].min()

        xz = mask.max(1)
        #print(xz.shape)
        xz_idx = np.where(xz==1)
        x2max,x2min = xz_idx[0].max(), xz_idx[0].min()
        z2max,z2min = xz_idx[1].max(), xz_idx[1].min()
        
        return img[y1min:y1max, x1min:x1max, z2min:z2max]
    else:
        return np.where(mask, img, 0)
    
    
def plot_2_histograms(insight, rim, bins=128, figsize=(22,16), fontsize=25, title='Histograms of "insight" and "rim" images', legendloc=0):
    """
    Plots histograms of two images: insight and rim
    
    C: 2021.11.09 / U: 2021.11.09
    """
    # remove zeros from the images
    insight = insight[insight>0]
    rim = rim[rim>0]

    f, ax = plt.subplots(1,1,figsize=(22,16))
    _ = ax.hist(insight.flat[:], bins, alpha=0.5, label='insight', color='red')
    _ = ax.hist(rim.flat[:], bins, alpha=0.5, label='rim', color='green')
    ax.set_title(title, fontsize=fontsize, fontweight='bold')
    #ax.set_ylim(top=1000)
    ax.legend(fontsize=fontsize-4, loc=legendloc)
    
    plt.grid(True)
    plt.show()
    
    
def plot_2_histograms_separately(insight, rim, bins=128, figsize=(22,16), fontsize=29, title='Histograms of "insight" and "rim" images', legendloc=0):
    """
    Plots histograms of two images: insight and rim separately and together.
    
    C: 2021.11.10 / U: 2021.11.10
    """
    # remove zeros from the images
    insight = insight[insight>0]
    rim = rim[rim>0]

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    
    # hist 1
    ax1 = fig.add_subplot(gs[0,0])
    _ = ax1.hist(insight.flat[:], bins, alpha=0.5, label='insight', color='red')
    ax1.legend(fontsize=fontsize-4, loc=legendloc)
    ax1.set_title('Inside ROI', fontsize=fontsize-4)
    ax1.grid(True)
    
    # hist 2
    ax2 = fig.add_subplot(gs[0,1],sharex=ax1, sharey=ax1)
    _ = ax2.hist(rim.flat[:], bins, alpha=0.5, label='rim', color='green')
    ax2.legend(fontsize=fontsize-4, loc=legendloc)
    ax2.set_title('Inside rim', fontsize=fontsize-4)
    ax2.grid(True)
    
    # hist 1 and hist 2
    ax3 = fig.add_subplot(gs[1, :])
    _ = ax3.hist(insight.flat[:], bins, alpha=0.5, label='insight', color='red')
    _ = ax3.hist(rim.flat[:], bins, alpha=0.5, label='rim', color='green')
    ax3.legend(fontsize=fontsize-4, loc=legendloc)
    ax3.grid(True)
    
    #ax.set_ylim(top=1000)    
    plt.suptitle(title, fontsize=fontsize, fontweight='bold')
    plt.show()
    
    
def plot_3_histograms_separately(insight, rim, bla, notebook_nr, bins=128, figsize=(22,16), fontsize=29, title='Histograms of "insight" and "rim" images',
                                 legendloc=0, save=False, sub='ccx'):
    """
    Plots histograms of two images: insight, rim and bladder separately and together.
    
    C: 2021.11.10 / U: 2021.11. 26
    """
    # remove zeros from the images
    insight = insight[insight>0]
    rim = rim[rim>0]
    bla = bla[bla>0]

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(2, 3)
    
    
    roi_col = '#00FFFF' # cyan #'#ECEC00' #'#FFD64E'
    rim_col = '#FF00FF' #magenta
    bla_col = 'blue'
    
    # hist 1
    ax1 = fig.add_subplot(gs[0,0])
    _ = ax1.hist(insight.flat[:], bins, alpha=0.5, label='insight', color=roi_col)
    ax1.legend(fontsize=fontsize-4, loc=legendloc)
    ax1.set_title('Inside ROI', fontsize=fontsize-4)
    ax1.grid(True)
    
    # hist 2
    ax2 = fig.add_subplot(gs[0,1],sharex=ax1, sharey=ax1)
    _ = ax2.hist(rim.flat[:], bins, alpha=0.5, label='rim', color=rim_col)
    ax2.legend(fontsize=fontsize-4, loc=legendloc)
    ax2.set_title('Inside rim', fontsize=fontsize-4)
    ax2.grid(True)
    
    # hist 3
    ax2 = fig.add_subplot(gs[0,2],sharex=ax1, sharey=ax1)
    _ = ax2.hist(bla.flat[:], bins, alpha=0.5, label='bladder', color=bla_col)
    ax2.legend(fontsize=fontsize-4, loc=legendloc)
    ax2.set_title('Inside bladder', fontsize=fontsize-4)
    ax2.grid(True)
    
    # hist 1 and hist 2
    ax3 = fig.add_subplot(gs[1, :])
    _ = ax3.hist(insight.flat[:], bins, alpha=0.5, label='insight', color=roi_col)
    _ = ax3.hist(rim.flat[:], bins, alpha=0.5, label='rim', color=rim_col)
    _ = ax3.hist(bla.flat[:], bins, alpha=0.5, label='bla', color=bla_col)
    
    ax3.legend(fontsize=fontsize-4, loc=legendloc)
    ax3.grid(True)
    
    #ax.set_ylim(top=1000)    
    plt.suptitle(title, fontsize=fontsize, fontweight='bold')
    
    if save:
        tit = f'../data/plots/{notebook_nr}-{sub}-{title}.png' 
        print(f'Saved figure:\n\t{tit}')
        plt.savefig(tit)
    plt.show()
    
    
def get_voxel_resolution_from_header(nii, name="Image"):
    """
    Get image resolution and voxel dimesnion based on the header. 
    
    Parameters:
    --------------------
    nii - structure of a Nifti image
    name - name of image (to print as info)
    
    
    Return:
    -------------------------------
    img_res - image resolution,
    vox_size - voxel size.
    
    
    C: 2021.11.08 / U:2021.11.08
    """
    hdr = nii.header
    
    img_res = hdr['dim'][1:4]
    vox_size = hdr['pixdim'][1:4]
    print(f'*** {name.upper()} ***')
    print(f'\tImage resolution: {img_res}')
    print(f'\tVoxel size: {vox_size}')
    return img_res, vox_size
    
    