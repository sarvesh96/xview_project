import sys
import cv2
import glob
import json
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.autograd import Variable

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _use_shared_memory

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.modules.loss import _assert_no_grad


class image_dataset(Dataset):
    def __init__(self, images_filename=None, boxes_filename=None, classes_filename=None, chip_dir_name=None, is_chipped=True):
        if is_chipped:
            self.images = np.load(images_filename, encoding='bytes')
            self.boxes = np.load(boxes_filename, encoding='bytes')
            self.classes = np.load(classes_filename, encoding='bytes')
        else:
            file_names = glob.glob(images_filename + "*.tif")
            file_names.sort()

            with open(boxes_filename) as f:
                json_data = json.load(f)

            self.coords = np.zeros((len(json_data['features']), 4))
            self.chips = np.zeros((len(json_data['features'])), dtype="object")
            self.classes = np.zeros((len(json_data['features'])))

            self.get_labels(json_data)

            self.images, self.boxes, classes = [], [], []
            for filename in tqdm(file_names[:10]):
                img = np.array(Image.open(filename))
                img_name = filename.split("/")[-1]

                img_coords = self.coords[self.chips == img_name]
                img_classes = self.classes[self.chips == img_name]
                img, box, clas = self.chip_image(img, img_coords, img_classes)
                # print('filename: {}'.format(filename), img.shape, box.shape, clas.shape)

                self.images.extend(list(img))
                self.boxes.extend(list(box))
                classes.extend(list(clas))

            self.images = np.array(self.images)
            self.boxes = np.array(self.boxes)
            self.classes = np.array(classes)

            np.save(chip_dir_name + 'images_600_num_10.npy', self.images)
            np.save(chip_dir_name + 'classes_600_num_10.npy', self.classes)
            np.save(chip_dir_name + 'boxes_600_num_10.npy', self.boxes)


    def __getitem__(self, index):
        return self.images[index], self.boxes[index], self.classes[index]

    def __len__(self):
        return self.images.shape[0]

    def get_labels(self, json_data):
        """
        Gets label data from a geojson label file and stores three arrays: coords, chips,
            and classes corresponding to the coordinates, file-names, and classes for
            each ground truth.

        Args:
            json_data: json file object to an xView geojson label file

        Output:
            None
        """
        for idx, features in enumerate(tqdm(json_data['features'])):
            if features['properties']['bounds_imcoords']:
                b_id = features['properties']['image_id']
                val = np.array([int(num) for num in features['properties']['bounds_imcoords'].split(",")])

                if val.shape[0] == 4:
                    self.coords[idx] = val
                self.chips[idx] = b_id
                self.classes[idx] = features['properties']['type_id']
            else:
                self.chips[idx] = 'None'

    @staticmethod
    def shuffle_images_and_boxes_classes(im, box, cls):
        """
        Shuffles images, boxes, and classes, while keeping relative matching indices
        Args:
            im: an array of images
            box: an array of bounding box coordinates ([xmin,ymin,xmax,ymax])
            cls: an array of classes
        Output:
            Shuffle image, boxes, and classes arrays, respectively
        """
        assert len(im) == len(box)
        assert len(box) == len(cls)

        perm = np.random.permutation(len(im))

        return im[perm], box[perm], cls[perm]

    def chip_image(self, image, coords, classes, shape=(600, 600)):
        """
        Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
            multiple chips are clipped: each portion that is in a chip is labeled. For example,
            half a building will be labeled if it is cut off in a chip. If there are no boxes,
            the boxes array will be [[0,0,0,0]] and classes [0].
            Note: This chip_image method is only tested on xView data-- there are some
                image manipulations that can mess up different images.
        Args:
            image: the image to be chipped in array format
            coords: an (N,4) array of bounding box coordinates for that image
            classes: an (N,1) array of classes for each bounding box
            shape: an (W,H) tuple indicating width and height of chips
        Output:
            An image array of shape (M,W,H,C), where M is the number of chips,
            W and H are the dimensions of the image, and C is the number of color
            channels.  Also returns boxes and classes dictionaries for each corresponding chip.
        """
        height, width, _ = image.shape
        wn, hn = shape

        w_num, h_num = (int(width / wn), int(height / hn))
        # images = np.zeros((w_num * h_num, hn, wn, 3))
        images = []
        total_boxes = []
        total_classes = []

        k = 0
        for i in range(w_num):
            for j in range(h_num):
                x = np.logical_or(np.logical_and((coords[:, 0] < ((i + 1) * wn)), (coords[:, 0] > (i * wn))),
                                  np.logical_and((coords[:, 2] < ((i + 1) * wn)), (coords[:, 2] > (i * wn))))
                out = coords[x]
                y = np.logical_or(np.logical_and((out[:, 1] < ((j + 1) * hn)), (out[:, 1] > (j * hn))),
                                  np.logical_and((out[:, 3] < ((j + 1) * hn)), (out[:, 3] > (j * hn))))
                out_n = out[y]
                out = np.transpose(np.vstack((np.clip(out_n[:, 0] - (wn * i), 0, wn),
                                              np.clip(out_n[:, 1] - (hn * j), 0, hn),
                                              np.clip(out_n[:, 2] - (wn * i), 0, wn),
                                              np.clip(out_n[:, 3] - (hn * j), 0, hn))))
                box_classes = classes[x][y]

                if out.shape[0]:
                    total_boxes.append(np.array(out))
                    total_classes.append(np.array(box_classes))
                    chip = image[hn * j:hn * (j + 1), wn * i:wn * (i + 1), :3]
                    images.append(chip)
                # else:
                #     total_boxes[k] = np.array([[0, 0, 0, 0]])
                #     total_classes[k] = np.array([0])

                k = k + 1

        images = np.array(images)
        total_boxes = np.array(total_boxes)
        total_classes = np.array(total_classes)
        image_tuple = (images.astype(np.uint8), total_boxes, total_classes)

        # print(images.shape, total_boxes.shape, total_classes.shape)

        return self.shuffle_images_and_boxes_classes(*image_tuple)


def plot_image(dataset, chip_image_index):
    fig, ax = plt.subplots(1)
    image, boxes, _ = dataset.__getitem__(chip_image_index)
    ax.imshow(image)

    # Create a Rectangle patch
    for k in range(len(boxes)):
        x_min = boxes[k][0]
        y_min = boxes[k][1]
        x_max = boxes[k][2]
        y_max = boxes[k][3]
        rect = patches.Rectangle((x_min, y_min), x_max-x_min,
                                  y_max-y_min, linewidth=1,
                                  edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir_name", default="../Data/train_images/",
                        help="Path to folder containing image chips \
                        (ie 'xview/train_images/' ")
    parser.add_argument("--json_file_path", default="../Data/xView_train.geojson",
                        help="File path to GEOJSON coordinate file")
    parser.add_argument("--images_filename", default="../Data/chipped/images.npy",
                        help="File path to images.npy file")
    parser.add_argument("--boxes_filename", default="../Data/chipped/boxes.npy",
                        help="File path to boxes.npy file")
    parser.add_argument("--classes_filename", default="../Data/chipped/classes.npy",
                        help="File path to classes.npy file")
    parser.add_argument("--chip_image_dir_name", default="../Data/chipped/",
                        help="File path to chipped image files")
    parser.add_argument("--is_chipped", default=False,
                        help="File path to chipped image files")
    args = parser.parse_args()

    if args.is_chipped:
        dataset = image_dataset(args.images_filename, args.boxes_filename,
                    args.classes_filename, is_chipped=args.is_chipped)
    else:
        dataset = image_dataset(args.image_dir_name, args.json_file_path,
                    chip_dir_name=args.chip_image_dir_name,
                    is_chipped=args.is_chipped)
