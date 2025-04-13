import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import random
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


ia.seed(1)
seq = iaa.Sequential([iaa.Sharpen((0.0, 1.0)),
                      iaa.Affine(scale=(1, 2)), iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Crop(percent=(0, 0.1))], random_order=True)



class Data(data.Dataset):
    def __init__(self, base_dir='./data/', train=True, dataset='ISIC2016', crop_szie=None):
        super(Data, self).__init__()
        self.dataset_dir = base_dir
        self.train = train
        self.dataset = dataset
        self.images = []
        self.labels = []


        if self.dataset == 'acdc' or self.dataset == 'synapse':
            self.crop_size = crop_szie
            if train:
                self.dir = os.path.join(self.dataset_dir, self.dataset + '/data_npz')
                txt = os.path.join(self.dataset_dir, self.dataset + '/annotations' + '/train.txt')
            else:
                self.dir = os.path.join(self.dataset_dir, self.dataset + '/data_npz')
                txt = os.path.join(self.dataset_dir, self.dataset + '/annotations' + '/test.txt')

            with open(txt, "r") as f:
                self.filename_list = f.readlines()
            for filename in self.filename_list:
                if self.dataset == 'acdc':
                    npz = os.path.join(self.dir, filename.strip() + '.npz')
                    data = np.load(npz)
                    image, label = data['image'], data['label']
                else:
                    npz = os.path.join(self.dir, filename.strip() + '.npz')
                    data = np.load(npz)
                    image, label = data['image'], data['label']

                image = np.array(image)
                label = np.array(label)

                if not self.train:
                    image = cv2.resize(image, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)
                    image = np.expand_dims(image, axis=2)
                    label = cv2.resize(label, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)

                self.images.append(image)
                self.labels.append(label)
                self.names.append(filename.strip())
            assert (len(self.images) == len(self.labels))


        if self.dataset == 'ISIC16' or self.dataset == 'ISIC18':
            if crop_szie is None:
                crop_szie = [512, 512]
            self.crop_size = crop_szie
            if train:
                self.image_dir = os.path.join(self.dataset_dir, self.dataset + '/images')
                self.label_dir = os.path.join(self.dataset_dir, self.dataset + '/labels')
                txt = os.path.join(self.dataset_dir, self.dataset + '/annotations' + '/train.txt')
            else:
                self.image_dir = os.path.join(self.dataset_dir, self.dataset + '/images')
                self.label_dir = os.path.join(self.dataset_dir, self.dataset + '/labels')
                txt = os.path.join(self.dataset_dir, self.dataset + '/annotations' + '/test.txt')

            with open(txt, "r") as f:
                self.filename_list = f.readlines()
            for filename in self.filename_list:
                image = os.path.join(self.image_dir, filename.strip() + '.jpg')
                image = Image.open(image)
                image = np.array(image)

                if self.dataset == 'ISIC16':
                    label = os.path.join(self.label_dir, filename.strip() + '.png')
                    label = Image.open(label)
                    label = np.array(label)
                if self.dataset == 'ISIC18':
                    label = os.path.join(self.label_dir, filename.strip() + '_segmentation.png')
                    label = Image.open(label)
                    label = np.array(label)

                if not self.train:
                    image = cv2.resize(image, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)
                    if self.dataset == 'ISIC16' or self.dataset == 'ISIC18':
                        label = cv2.resize(label, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST) / 255


                self.images.append(image)
                self.labels.append(label)
                self.names.append(filename.strip())

            assert(len(self.images) == len(self.labels))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {'image': self.images[index], 'label': self.labels[index]}

        prob = random.random()
        if self.dataset == 'acdc' or self.dataset == 'synapse':
            sample['label'] = np.array(sample['label']).astype(np.int16)
        if self.train and prob > 0.5:
            segmap = SegmentationMapsOnImage(sample['label'], shape=sample['image'].shape)
            sample['image'], sample['label'] = seq(image=sample['image'], segmentation_maps=segmap)
            sample['label'] = sample['label'].get_arr()

        if self.train:
            sample['image'] = cv2.resize(sample['image'], (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)
            if self.dataset == 'acdc' or self.dataset == 'synapse':
                sample['image'] = np.expand_dims(sample['image'], axis=2)
                
            if self.dataset == 'ISIC16' or self.dataset == 'ISIC18':
                sample['label'] = cv2.resize(sample['label'], (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST) / 255
            if self.dataset == 'acdc' or self.dataset == 'synapse':
                sample['label'] = np.array(sample['label']).astype("float")
                sample['label'] = cv2.resize(sample['label'], (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)

        return sample


    def __str__(self):
        return 'dataset:{} train:{}'.format(self.dataset, self.train)