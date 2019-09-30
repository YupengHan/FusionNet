import os
import numpy as np
from .util import read_image

class ImageKITTIDataset:
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
    :obj:`KITTI_LABEL_NAMES`.
    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.
    The type of the image, the bounding boxes and the labels are as follows.
    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`depth.dtype == numpy.float32`
    * :obj:`y_rot.dtype == numpy.float32`
    * :obj:`difficult.dtype == numpy.bool`
    '''
    def __init__(self, data_dir):
        id_list_file = os.path.join(data_dir, 'id_list_file.txt')
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = KITTI_LABEL_NAMES
        
    def __len__(self):
        return len(self.ids)
    def get_example(self, i):
        """Returns the i-th example.
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.
        Args:
            i (int): The index of the example.
        Returns:
            tuple of an image and bounding boxes
        """
        id_ = self.ids[i]
        bbox = list()
        label = list()
        difficult = list()
        depth = list()
        y_rot = list()
        
        label_f = os.path.join(self.data_dir, 'label_2', id_ + '.txt')
        lines = open(label_f).readlines()
        items = [x.strip(' ').split(' ') for x in lines]
        for i in range(len(lines)):
            name = items[i][0]
            '''
            ingore the DontCare part
            '''
            if name == 'DontCare':
                continue
            xmin, ymin, xmax, ymax = items[i][4:8]
            bbox.append([int(float(ymin)), int(float(xmin)), int(float(ymax)), int(float(xmax))])
            label.append(KITTI_LABEL_NAMES.index(name))
            difficult.append(False)
            
            depth_ = float(items[i][13])/70.0
            if abs(depth_) > 1:
                depth_ = 1
            depth.append(depth_)
            
            y_rot.append(float(items[i][3]))



        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        depth = np.stack(depth).astype(np.float32)
        y_rot = np.stack(y_rot).astype(np.float32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'image_2', id_ + '.png')
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult, depth, y_rot
    __getitem__ = get_example

KITTI_LABEL_NAMES = (
    'Car',
    'Pedestrian',
    'Cyclist',
    'Truck',
    'Misc',
    'Van',
    'Tram',
    'Person_sitting',
    'DontCare',)
