import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

def crop(ct, seg):
    x, y, z = np.where(seg)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    return ct[x_min:x_max, y_min:y_max, z_min:z_max],\
            seg[x_min:x_max, y_min:y_max, z_min:z_max]

def padding(ct):
    newimg = np.zeros((224, 160, 224))
    ct_size = ct.shape
    offset = (np.array(newimg.shape) - np.array(ct_size)) // 2
    newimg[offset[0]: offset[0] + ct_size[0], 
           offset[1]: offset[1] + ct_size[1], 
           offset[2]: offset[2] + ct_size[2]] = ct
    return newimg

if __name__=='__main__':
    ct_path = 'train/volume-%d.nii'
    seg_path = 'train/segmentation-%d.nii'
    l = []
    for i in range(28, 131):
        ct = nib.load(ct_path % i).get_data()
        seg = nib.load(seg_path % i).get_data()
        spacing = sitk.ReadImage(ct_path % i).GetSpacing()
        newspace = np.array(spacing) / np.array([2., 2., 2.])
        ct, seg = crop(ct, seg)
        ct = zoom(ct, (1, 1, spacing[2]), mode = 'nearest',order=3)
        seg = zoom(seg, (1, 1, spacing[2]), mode = 'nearest',order=3)
        ct = ct[::2, ::2, ::2]
        seg = seg[::2, ::2, ::2]
        ct = padding(ct)
        seg = padding(seg)
        ct = ct.clip(-200, 250)
        np.save('preprocess_3d/volume-%d.npy' % i, ct)
        np.save('preprocess_3d/segmentation-%d.npy' % i, seg)
        l.append(ct.shape)
    l = np.array(l).transpose(1,0)
    print np.max(l, axis=1)
    plt.show(plt.hist(l[0],40))

        
