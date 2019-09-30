from __future__ import  absolute_import
from __future__ import  division
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt
from math import pi
from data.kitti_img_dataset import ImageKITTIDataset

def inverse_normalize(img):
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()

def preprocess(img, min_size=375, max_size=1242):
    """Preprocess an image for feature extraction.
    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.
    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.
    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
    Returns:
        ~numpy.ndarray: A preprocessed image.
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=375, max_size=1242):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label, depth, y_rot = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])
        if params['x_flip']:
            for i in range(len(y_rot)):
                theta = float(y_rot[i])
                if theta > 0:
                    y_rot[i] = pi - theta
                if theta < 0:
                    y_rot[i] = -pi - theta
                    y_rot[i]
                if theta == 0:
                    y_rot[i] = 3.14

        return img, bbox, label, depth, y_rot, scale



'''
    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.
    
    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.
    
    The type of the image, the bounding boxes and the labels are as follows.
    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

'''


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = ImageKITTIDataset(opt.train_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult, depth, y_rot = self.db.get_example(idx)
        '''
        If :obj:`return_difficult == True`, this dataset returns corresponding
        :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
        that indicates whether bounding boxes are labeled as difficult or not.
        '''
        img, bbox, label, depth, y_rot, scale = self.tsf((ori_img, bbox, label, depth, y_rot))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), depth.copy(), y_rot.copy(), scale

    def __len__(self):
        return len(self.db)
    

class TestDataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = ImageKITTIDataset(opt.test_data_dir)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult, depth, y_rot = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
