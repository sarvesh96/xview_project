import os
import sys
import time
import torch
import argparse
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn as cudnn

from data import *
from ssd import build_ssd
from torch.autograd import Variable
from layers.modules import MultiBoxLoss
from utils.augmentations import SSDAugmentation
from torch.utils.data.dataloader import DataLoader


def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
	description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='XVIEW', choices=['XVIEW'],
					type=str, help='XVIEW')
parser.add_argument('--images_filename', default='../../Data/chipped/images_300_num_25.npy',
					type=str, help='location of _images.npy')
parser.add_argument('--boxes_filename', default='../../Data/chipped/boxes_300_num_25.npy',
					type=str, help='location of _boxes.npy')
parser.add_argument('--classes_filename', default='../../Data/chipped/classes_300_num_25.npy',
					type=str, help='location of _classes.npy')
parser.add_argument('--dataset_root', default=XVIEW_ROOT,
					help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
					help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
					help='Batch size for training')
parser.add_argument('--num_epochs', default=10, type=int,
					help='Number of epochs on the data')
parser.add_argument('--resume', default=None, type=str,
					help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
					help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
					help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
					help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
					help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
					help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
					help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
					help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
					help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
					help='Directory for saving checkpoint models')
parser.add_argument('--verbose', default=True,
					help='')
args = parser.parse_args()


if torch.cuda.is_available():
	if args.cuda:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
	if not args.cuda:
		print("WARNING: It looks like you have a CUDA device, but aren't " +
			  "using CUDA.\nRun with --cuda for optimal training speed.")
		torch.set_default_tensor_type('torch.FloatTensor')
else:
	torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
	os.mkdir(args.save_folder)


def train():
	cfg = xview

	if args.verbose:
		print('Loading Dataset...')
	dataset = XVIEWDetection(args.images_filename,
							args.boxes_filename,
							args.classes_filename,
							transform=SSDAugmentation(cfg['min_dim'],
								MEANS))

	# if args.visdom:
	#     import visdom
	#     viz = visdom.Visdom()

	if args.verbose:
		print('Building Model...')
	ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
	net = ssd_net

	if args.cuda:
		net = torch.nn.DataParallel(ssd_net)
		cudnn.benchmark = True

	if args.verbose:
		print('Resuming/Loading-New...')
	if args.resume:
		print('Resuming training, loading {}...'.format(args.resume))
		ssd_net.load_weights(args.resume)
	else:
		vgg_weights = torch.load(args.save_folder + args.basenet)
		print('Loading base network...')
		ssd_net.vgg.load_state_dict(vgg_weights)

	if args.cuda:
		net = net.cuda()

	if not args.resume:
		print('Initializing weights...')
		# initialize newly added layers' weights with xavier method
		ssd_net.extras.apply(weights_init)
		ssd_net.loc.apply(weights_init)
		ssd_net.conf.apply(weights_init)

	if args.verbose:
		print('Loading Optimizer & Criterion...')
	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
						  weight_decay=args.weight_decay)
	criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
							 False, args.cuda)

	net.train()
	# loss counters
	loc_loss = 0
	conf_loss = 0
	epoch = 0
	print('Loading the dataset...')

	epoch_size = len(dataset) // args.batch_size
	print('Training SSD on:', dataset.name)
	print('Using the specified args:')


	step_index = 0

	# if args.visdom:
	#     vis_title = 'SSD.PyTorch on ' + dataset.name
	#     vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
	#     iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
	#     epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

	data_loader = DataLoader(dataset, args.batch_size,
							  num_workers=args.num_workers,
							  shuffle=True, collate_fn=detection_collate,
							  pin_memory=True)
	
	# create batch iterator
	for epoch_ in range(args.num_epochs):
		batch_iterator = iter(data_loader)
		for iteration in range(args.start_iter, cfg['max_iter']):
			# if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
			#     update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
			#                     'append', epoch_size)
			#     # reset epoch loss counters
			#     loc_loss = 0
			#     conf_loss = 0
			#     epoch += 1

			if iteration in cfg['lr_steps']:
				step_index += 1
				adjust_learning_rate(optimizer, args.gamma, step_index)

			# load train data
			images, targets = next(batch_iterator)

			if args.cuda:
				images = Variable(images.cuda())
				targets = [Variable(ann.cuda(), requires_grad=True) for ann in targets]
			else:
				images = Variable(images)
				targets = [Variable(ann, requires_grad=True) for ann in targets]
			# forward
			t0 = time.time()
			out = net(images)
			# backprop
			optimizer.zero_grad()
			loss_l, loss_c = criterion(out, targets)
			loss = loss_l + loss_c
			print('Loc Loss:{:>10.4f}| Conf Loss:{:10.4f}'.format(loss_l, loss_c))
			loss.backward()
			optimizer.step()
			t1 = time.time()
			loc_loss += loss_l.data
			conf_loss += loss_c.data

			if iteration % 10 == 0:
				print('timer: %.4f sec.' % (t1 - t0))
				print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data))#, end=' ')

			# if args.visdom:
			#     update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
			#                     iter_plot, epoch_plot, 'append')

			if iteration != 0 and iteration % 5000 == 0:
				print('Saving state, iter:', iteration)
				torch.save(ssd_net.state_dict(), 'weights/ssd300_XVIEW_' +
						   repr(iteration) + '.pth')
		torch.save(ssd_net.state_dict(),
				   args.save_folder + args.dataset + '_' + str(epoch_) + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
	"""Sets the learning rate to the initial LR decayed by 10 at every
		specified step
	# Adapted from PyTorch Imagenet example:
	# https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	lr = args.lr * (gamma ** (step))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def xavier(param):
	init.xavier_uniform_(param)


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		xavier(m.weight.data)
		m.bias.data.zero_()


# def create_vis_plot(_xlabel, _ylabel, _title, _legend):
#     return viz.line(
#         X=torch.zeros((1,)).cpu(),
#         Y=torch.zeros((1, 3)).cpu(),
#         opts=dict(
#             xlabel=_xlabel,
#             ylabel=_ylabel,
#             title=_title,
#             legend=_legend
#         )
#     )


# def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
#                     epoch_size=1):
#     viz.line(
#         X=torch.ones((1, 3)).cpu() * iteration,
#         Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
#         win=window1,
#         update=update_type
#     )
#     # initialize epoch plot on first iteration
#     if iteration == 0:
#         viz.line(
#             X=torch.zeros((1, 3)).cpu(),
#             Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
#             win=window2,
#             update=True
#         )


if __name__ == '__main__':
	train()
