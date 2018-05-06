import os
import os.path as osp
import numpy as np

from .util import read_image

HOME = '/home/ubuntu/Project/xview_project/frcnn/'

def map_labels_contiguous(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class XVIEWBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`XVIEW_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data.
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        images_filename = '/home/ubuntu/Project/Data/chipped/images_600_num_10.npy'
        boxes_filename = '/home/ubuntu/Project/Data/chipped/boxes_600_num_10.npy'
        classes_filename = '/home/ubuntu/Project/Data/chipped/classes_600_num_10.npy'

        self.images = np.load(images_filename, encoding='bytes')
        self.boxes = np.load(boxes_filename, encoding='bytes')
        self.classes = np.load(classes_filename, encoding='bytes')

        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = XVIEW_BBOX_LABEL_NAMES
        self.label_map = map_labels_contiguous(osp.join(HOME, 'data/xview_labels.txt'))

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        img = self.images[i].transpose((2, 0, 1)).astype(dtype=np.float32)
        bbox = self.boxes[i].astype(dtype=np.float32)
        img_class_xview = self.classes[i]
        label = np.array([[self.label_map[int(x)]] if x in self.label_map else [self.label_map[0]]for x in img_class_xview]).astype(dtype=np.int32).reshape(-1)

        difficult = np.zeros(bbox.shape[0], dtype=np.bool).astype(np.uint8)
        return img, bbox, label, difficult

    __getitem__ = get_example


XVIEW_BBOX_LABEL_NAMES = ('__noclass__', 'fixed_wing_aircraft', 'small_aircraft', 'passenger_OR_cargo_plane', 'helicopter',
               'passenger_vehicle', 'small_car', 'bus', 'pickup_truck', 'utility_truck', 'truck', 'cargo_truck',
               'truck_tractor_with_box_trailer', 'truck_tractor', 'trailer', 'truck_tractor_with_flatbed_trailer',
               'truck_tractor_with_liquid_tank', 'crane_truck', 'railway_vehicle', 'passenger_car',
               'cargo_OR_container_car', 'flat_car', 'tank_car', 'locomotive', 'maritime_vessel', 'motorboat',
               'sailboat', 'tugboat', 'barge', 'fishing_vessel', 'ferry', 'yacht', 'container_ship', 'oil_tanker',
               'engineering_vehicle', 'tower_crane', 'container_crane', 'reach_stacker', 'straddle_carrier',
               'mobile_crane', 'dump_truck', 'haul_truck', 'tractor', 'front_loader_OR_bulldozer', 'excavator',
               'cement_mixer', 'ground_grader', 'hut_OR_tent', 'shed', 'building', 'aircraft_hangar',
               'damaged_OR_demolished_building', 'facility', 'construction_site', 'vehicle_lot',
               'helipad,storage_tank', 'shipping_container_lot', 'shipping_container', 'pylon', 'tower')
