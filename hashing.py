import argparse
from torch.utils.data import DataLoader
import os
from model.dpsh import *
from model.dph import *
from model.hashnet import *
from model.psldh import *
from utils.data_provider import *


# os.environ["CUDA_VISIBLE_DEVICES"]='7'

parser = argparse.ArgumentParser()
# description of data
parser.add_argument('--dataset_name', dest='dataset', default='CIFAR-10', choices=['CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'], help='name of the dataset')
parser.add_argument('--data_dir', dest='data_dir', default='/home/trc/datasets/dataset_mat/cifar.h5', help='path of the dataset')
parser.add_argument('--database_file', dest='database_file', default='database_img.txt', help='the image list of database images')
parser.add_argument('--train_file', dest='train_file', default='train_img.txt', help='the image list of training images')
parser.add_argument('--test_file', dest='test_file', default='test_img.txt', help='the image list of test images')
parser.add_argument('--database_label', dest='database_label', default='database_label.txt', help='the label list of database images')
parser.add_argument('--train_label', dest='train_label', default='train_label.txt', help='the label list of training images')
parser.add_argument('--test_label', dest='test_label', default='test_label.txt', help='the label list of test images')
# model
parser.add_argument('--hashing_method', dest='method', default='DPH', choices=['DPH', 'DPSH', 'HashNet', 'PSLDH'], help='deep hashing methods')
parser.add_argument('--backbone', dest='backbone', default='VGG11', choices= ['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50'], help='backbone network')
parser.add_argument('--yita', dest='yita', type=int, default=50, help='yita in the dpsh paper')
parser.add_argument('--code_length', dest='bit', type=int, default=64, help='length of the hashing code')
# training or test
parser.add_argument('--train', dest='train', type=bool, default=True, choices=[True, False], help='to train or not')
parser.add_argument('--test', dest='test', type=bool, default=True, choices=[True, False], help='to test or not')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='number of images in one batch')
parser.add_argument('--load_model', dest='load', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='save', default='checkpoint/', help='models are saved here')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=150, help='number of epoch')
parser.add_argument('--round', dest='round', type=int, default=1, help='round')
parser.add_argument('--gpuid', dest='gpuid', type=int, default=1, help='gpuid')
parser.add_argument('--learning_rate', dest='lr', type=float, default=0.01, help='initial learning rate for sgd')
parser.add_argument('--weight_decay', dest='wd', type=float, default=1e-5, help='weight decay for SGD')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
dset_database = HashingDataset(args.data_dir, 'dataset')
dset_train = HashingDataset(args.data_dir, 'train')
dset_test = HashingDataset(args.data_dir, 'test')
num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)

database_loader = DataLoader(dset_database, batch_size=args.batch_size, shuffle=False, num_workers=4)
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

database_labels = dset_database.label
train_labels = dset_train.label
test_labels = dset_test.label
os.makedirs(args.save, exist_ok=True)
model = None
if args.method == 'DPSH':
    model = DPSH(args)
    if args.train:
        model.train(train_loader, train_labels, num_train, database_loader, test_loader, database_labels, test_labels, num_database, num_test)
elif args.method == 'PSLDH':
    model = PSLDH(args)
    if args.train:
        model.train(train_loader, train_labels, num_train, database_loader, test_loader, database_labels, test_labels,
                    num_database, num_test)

if args.test:
    model.load_model()
    model.test(database_loader, test_loader, database_labels, test_labels, num_database, num_test)
