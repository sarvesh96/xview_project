# TODO: Change HOME Path
import PIL
import cv2
import sys
import torch
import numpy as np
import os.path as osp

from .config import HOME
from torch.utils.data.dataset import Dataset


XVIEW_CLASSES = (  # always index 0, total of 94 classes
	'__NOCLASS__', '__NOCLASS__', '__NOCLASS__',
	'__NOCLASS__', '__NOCLASS__', '__NOCLASS__',
	'__NOCLASS__', '__NOCLASS__', '__NOCLASS__',
	'__NOCLASS__', '__NOCLASS__', 'fixed_wing_aircraft',
	'small_aircraft', 'passenger_OR_cargo_plane', '__NOCLASS__',
	'helicopter', '__NOCLASS__', 'passenger_vehicle', 'small_car',
	'bus', 'pickup_truck', 'utility_truck', '__NOCLASS__',
	'truck', 'cargo_truck', 'truck_tractor_with_box_trailer',			# 25 classes
	'truck_tractor', 'trailer', 'truck_tractor_with_flatbed_trailer',
	'truck_tractor_with_liquid_tank', '__NOCLASS__', '__NOCLASS__',
	'crane_truck', 'railway_vehicle', 'passenger_car',
	'cargo_OR_container_car', 'flat_car', 'tank_car', 'locomotive',
	'__NOCLASS__', 'maritime_vessel', 'motorboat', 'sailboat',
	'__NOCLASS__', 'tugboat', 'barge', '__NOCLASS__',
	'fishing_vessel', '__NOCLASS__', 'ferry', 'yacht',					# 50 classes
	'container_ship', 'oil_tanker', 'engineering_vehicle',
	'tower_crane', 'container_crane', 'reach_stacker',
	'straddle_carrier', '__NOCLASS__', 'mobile_crane', 'dump_truck',
	'haul_truck', 'tractor', 'front_loader_OR_bulldozer', 'excavator',
	'cement_mixer', 'ground_grader', '__NOCLASS__', '__NOCLASS__',
	'__NOCLASS__', '__NOCLASS__', 'hut_OR_tent', 'shed',
	'building', 'aircraft_hangar', '__NOCLASS__',						# 75 classes
	'damaged_OR_demolished_building', 'facility', '__NOCLASS__',
	'construction_site', '__NOCLASS__', '__NOCLASS__', '__NOCLASS__',
	'vehicle_lot', 'helipad', '__NOCLASS__', 'storage_tank',
	'__NOCLASS__', '__NOCLASS__', 'shipping_container_lot',
	'__NOCLASS__', 'shipping_container', '__NOCLASS__', 'pylon', 'tower')


# note: if you used our download scripts, this should be right
XVIEW_ROOT = osp.join(HOME, "data/")


def map_labels_contiguous(label_file):
	label_map = {}
	labels = open(label_file, 'r')
	for line in labels:
		ids = line.split(',')
		label_map[int(ids[0])] = int(ids[1])
	return label_map


class XVIEWAnnotationTransform(object):
	"""Transforms a Xview annotation into a Tensor of bbox coords and label index
	Initilized with a dictionary lookup of classnames to indexes
	Arguments:
		class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
			(default: alphabetic indexing of Xview's 60 classes)
	"""

	def __init__(self):
		self.label_map = map_labels_contiguous(osp.join(XVIEW_ROOT, 'xview_labels.txt'))

	def __call__(self, bounding_boxes, img_class_xview):
		"""
		Arguments:
			bounding_boxes (np.ndarray): 2D array containing [xmin, ymin, xmin, xmax] of multiple
										classes identified in the given image
			img_class_xview (np.ndarray): 1D array containing class ID of the respective boxes
		Returns:
			2D np.ndarray containing lists of bounding boxes and labels associated [bbox coords, class name]
		"""
		img_class = np.array([[self.label_map[x] for x in img_class_xview]])
		res = np.hstack((bounding_boxes, np.expand_dims(img_class, axis=1)))

		return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class XVIEWDetection(Dataset):
	"""XVIEW Detection Dataset Object
	input is image, target is annotation
	Arguments:
		root (string): filepath to XVIEW folder.
		image_set (string): imageset to use (eg. 'train', 'val', 'test')
		transform (callable, optional): transformation to perform on the
			input image
		target_transform (callable, optional): transformation to perform on the
			target `annotation`
			(eg: take in caption string, return tensor of word indices)
		dataset_name (string, optional): which dataset to load
			(default: 'XVIEW')
	"""

	def __init__(self, images_filename, boxes_filename, classes_filename,
				 transform=None, target_transform=XVIEWAnnotationTransform(),
				 dataset_name='XVIEW'):

		self.transform = transform
		self.target_transform = target_transform
		self.name = dataset_name

		self.images = np.load(images_filename, encoding='bytes')
		self.boxes = np.load(boxes_filename, encoding='bytes')
		self.classes = np.load(classes_filename, encoding='bytes')
	

	def __getitem__(self, index):
		img = self.images[index]

		if self.target_transform is not None:
			target = self.target_transform(self.boxes[index], self.classes[index])

		if self.transform is not None:
			target = np.array(target)
			img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
			# to rgb
			img = img[:, :, (2, 1, 0)]
			# img = img.transpose(2, 0, 1)
			target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

		return torch.from_numpy(img).permute(2, 0, 1), target


	def __len__(self):
		return self.images.shape[0]


	def pull_item(self, index):
		# Implemented as __getitem__
		self.__getitem__(index)


	def pull_image(self, index):
		'''Returns the original image object at index in PIL form
		Note: not using self.__getitem__(), as any transformations passed in
		could mess up this functionality.
		Argument:
			index (int): index of img to show
		Return:
			PIL img
		'''
		return cv2.imread(self.images[index], mode=None) # Mode will be determined from type if None


	def pull_anno(self, index):
		'''Returns the original annotation of image at index
		Note: not using self.__getitem__(), as any transformations passed in
		could mess up this functionality.
		Argument:
			index (int): index of img to get annotation of
		Return:
			list:  [img_id, [(label, bbox coords),...]]
				eg: ('001718', [('dog', (96, 13, 438, 332))])
		'''
		img_id = str(index)
		gt = self.target_transform(self.boxes[index], self.classes[index])

		return img_id, gt


	def pull_tensor(self, index):
		'''Returns the original image at an index in tensor form
		Note: not using self.__getitem__(), as any transformations passed in
		could mess up this functionality.
		Argument:
			index (int): index of img to show
		Return:
			tensorized version of img, squeezed
		'''
		return torch.Tensor(self.images[index].unsqueeze_(0))