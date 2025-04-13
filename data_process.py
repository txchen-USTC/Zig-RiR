import numpy as np
import nibabel as nb
import glob
import cv2
import h5py
import os
import random
from tqdm import tqdm

## the directory with datatset
root_dir = "training/*/*"
files = glob.glob(r"training/*/*")
labels = []
images = []
testval_list = random.sample(range(1, 101), 30)  #
testval_list = list(map(lambda x: str(x).zfill(3), testval_list))



test_list = random.sample(testval_list, 20)
for each in files:
    if "frame" in each and "gt" in each:
        labels.append(each)
    elif "frame" in each:
        images.append(each)

os.makedirs('npz', exist_ok=True)
os.makedirs('npz', exist_ok=True)
os.makedirs('h5', exist_ok=True)


last_patient = None
slice_num = 0

train_file = open('/data', 'w')
test_file = open('/data', 'w')
test_h5_file = open('/data', 'w')
for i in tqdm(range(len(images))):
    patient = images[i].split("\\")[-2]


    image = nb.load(images[i]).get_fdata()
    label = nb.load(labels[i]).get_fdata()
    size = image.shape
    if size[0] > size[1]:
        pad_left = (size[0] - size[1]) // 2 + (size[0] - size[1]) % 2
        pad_right = (size[0] - size[1]) // 2
        image = np.pad(image, ((0, 0), (pad_left, pad_right), (0, 0)), 'constant')
        label = np.pad(label, ((0, 0), (pad_left, pad_right), (0, 0)), 'constant')
    elif size[0] < size[1]:
        pad_up = (size[1] - size[0]) // 2 + (size[1] - size[0]) % 2
        pad_down = (size[1] - size[0]) // 2
        image = np.pad(image, ((pad_up, pad_down), (0, 0), (0, 0)), 'constant')
        label = np.pad(label, ((pad_up, pad_down), (0, 0), (0, 0)), 'constant')

    assert image.shape[0] == image.shape[1], 'padding failed'
    assert image.shape[2] == label.shape[2], f'{image.shape[2], label.shape[2], images[i], labels[i]}'
    slices = image.shape[2]
    if last_patient == patient:
        tag = '02'
    else:
        tag = '01'
        last_patient = patient
    if patient[-3:] not in testval_list:
        for num in range(slices):
            case_image = cv2.resize(image[:, :, num], (256, 256), interpolation=cv2.INTER_NEAREST)
            case_label = cv2.resize(label[:, :, num], (256, 256), interpolation=cv2.INTER_NEAREST)
            np.savez("npz/" + str(patient) + f"_{tag}" + "_slice" + str(num).zfill(3),
                     image=case_image, label=case_label)
            train_file.write(str(patient) + f"_{tag}" + "_slice" + str(num).zfill(3) + '\n')
            slice_num += 1

    if patient[-3:] in test_list:
        image_h5, label_h5 = np.zeros((slices, 256, 256)), np.zeros((slices, 256, 256))
        for num in range(slices):
            case_image = cv2.resize(image[:, :, num], (256, 256), interpolation=cv2.INTER_NEAREST)
            case_label = cv2.resize(label[:, :, num], (256, 256), interpolation=cv2.INTER_NEAREST)
            image_h5[num], label_h5[num] = case_image, case_label
            np.savez("npz/" + str(patient) + f"_{tag}" + "_slice" + str(num).zfill(3),
                     image=case_image, label=case_label)
            test_file.write(str(patient) + f"_{tag}" + "_slice" + str(num).zfill(3) + '\n')
            slice_num += 1
        with h5py.File(f"/npz", 'w') as f:
            f.create_dataset('image', data=image_h5)
            f.create_dataset('label', data=label_h5)
        test_h5_file.write(f'{patient}_{tag}.npy.h5' + '\n')


train_file.close()
test_file.close()
test_h5_file.close()