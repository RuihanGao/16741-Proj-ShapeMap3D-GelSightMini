import numpy as np
import scipy
from scipy import interpolate
from scipy.ndimage import gaussian_filter

def skewMat(v):
    mat = np.zeros((3,3))
    mat[0,1] = -1*v[2]
    mat[0,2] = v[1]

    mat[1,0] = v[2]
    mat[1,2] = -1*v[0]

    mat[2,0] = -1*v[1]
    mat[2,1] = v[0]

    return mat

def fill_blank(img):
    # here we assume there are some zero value holes in the image,
    # and we hope to fill these holes with interpolation
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    #mask invalid values
    array = np.ma.masked_where(img == 0, img)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = img[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                 method='linear', fill_value = 0) # cubic # nearest # linear
    return GD1

def padding(img):
    # pad one row & one col on each side
    if len(img.shape) == 2:
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')
    elif len(img.shape) == 3:
        return np.pad(img, ((1, 1), (1, 1), (0, 0)), 'symmetric')

def generate_normals(height_map):
    # height_map = cv2.GaussianBlur(height_map.astype(np.float32),(5,5),0)
    # height_map = cv2.GaussianBlur(height_map.astype(np.float32),(3,3),0)
    [h,w] = height_map.shape
    center = height_map[1:h-1,1:w-1] # z(x,y)
    top = height_map[0:h-2,1:w-1] # z(x-1,y)
    bot = height_map[2:h,1:w-1] # z(x+1,y)
    left = height_map[1:h-1,0:w-2] # z(x,y-1)
    right = height_map[1:h-1,2:w] # z(x,y+1)
    dzdx = (bot-top)/2.0
    dzdy = (right-left)/2.0
    direction = -np.ones((h-2,w-2,3))
    direction[:,:,0] = dzdx
    direction[:,:,1] = dzdy
    # direction[:,:,0] = -(bot-center)/1.0
    # direction[:,:,1] = -(right-center)/1.0

    mag_tan = np.sqrt(dzdx**2 + dzdy**2)
    # plt.imshow(mag_tan)
    # plt.show()
    grad_mag = np.arctan(mag_tan)
    invalid_mask = mag_tan == 0
    valid_mask = ~invalid_mask
    grad_dir = np.zeros((h-2,w-2))
    grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask]/mag_tan[valid_mask], dzdy[valid_mask]/mag_tan[valid_mask])
    # grad_dir[invalid_mask] = 0
    # grad_dir[valid_mask] = np.arccos(-dzdx[valid_mask]/mag_tan[valid_mask])

    magnitude = np.sqrt(direction[:,:,0]**2 + direction[:,:,1]**2 + direction[:,:,2]**2)
    normal = direction/magnitude[:,:, np.newaxis] # unit norm

    normal = padding(normal)
    grad_mag = padding(grad_mag)
    grad_dir = padding(grad_dir)
    return grad_mag, grad_dir, normal

def processInitialFrame(f_init):
    # gaussian filtering with square kernel with
    # filterSize : kscale*2+1
    # sigma      : kscale
    kscale = 50

    img_d = f_init.astype('float')
    convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)

    f0 = f_init.copy()
    for ch in range(img_d.shape[2]):
        f0[:,:, ch] = convEachDim(img_d[:,:,ch])

    frame_ = img_d

    # Checking the difference between original and filtered image
    diff_threshold = 5
    dI = np.mean(f0-frame_, axis=2)
    idx =  np.nonzero(dI<diff_threshold)

    # Mixing image based on the difference between original and filtered image
    frame_mixing_per = 0.15
    h,w,ch = f0.shape
    pixcount = h*w

    for ch in range(f0.shape[2]):
        f0[:,:,ch][idx] = frame_mixing_per*f0[:,:,ch][idx] + (1-frame_mixing_per)*frame_[:,:,ch][idx]

    return f0
