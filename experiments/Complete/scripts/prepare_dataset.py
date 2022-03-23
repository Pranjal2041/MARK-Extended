import requests
from tqdm import tqdm
import os
import zipfile
import numpy as np
import torchvision
import torchvision.transforms as transforms
import h5py

def download_dataset(dset_name, link, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Download the dataset from the given link with a progress bar and save it to the given directory
    r = requests.get(link, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(os.path.join(save_dir, f'{dset_name}.zip'), 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        raise Exception("ERROR, something went wrong")
    return os.path.join(save_dir, f'{dset_name}.zip')


def cifar_to_h5py(dset, filename, num_classes = 100, num_tasks = 20, val_split = 0.2, isTrain = True, classes_shuff = None):
    if classes_shuff is None:
        classes_shuff = np.random.permutation(np.arange(num_classes))
    label_to_task = {ii:i%num_tasks for i,ii in enumerate(classes_shuff)}
    label_to_pseudo_label = {classes_shuff[i]:i//20 for i in range(num_classes)}
    train_dset = [[] for _ in range(num_tasks)]
    for item in dset:
        lab = item[1]
        task_num = label_to_task[lab]
        train_dset[task_num].append(item)
    if isTrain:
        # Need to create the val file
        f_val = h5py.File(filename.replace('_train', '_val'),'w')
    f  = h5py.File(filename,'w')
    for task in range(20):
        X = np.stack([x[0] for x in train_dset[task]])
        Y = np.stack([label_to_pseudo_label[x[1]] for x in train_dset[task]])
        Y_orig = np.stack([x[1] for x in train_dset[task]])
        if isTrain:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            size = int(val_split*X.shape[0])
            X_val = X[indices[:size]]
            Y_val = Y[indices[:size]]
            Y_orig_val = Y_orig[indices[:size]]
            X = X[indices[size:]]
            Y = Y[indices[size:]]
            Y_orig = Y_orig[indices[size:]]
            gr_val = f_val.create_group(str(task))
            gr_val.create_dataset('X',data=X_val)
            gr_val.create_dataset('Y',data=Y_val)
            gr_val.create_dataset('Y_orig',data=Y_orig_val)

        gr = f.create_group(str(task))
        gr.create_dataset('X',data=X)
        gr.create_dataset('Y',data=Y)
        gr.create_dataset('Y_orig',data=Y_orig)
    f.close()
    if isTrain:
        f_val.close()

def prepare(dset, save_dir = 'datasets'):
    os.makedirs(save_dir, exist_ok = True)
    if dset['name'] == 'cifar100':
        np.random.seed(32112)
        classes_shuff = np.random.permutation(np.arange(100))
        tr_set = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=True, transform=transforms.ToTensor())
        cifar_to_h5py(tr_set, os.path.join(save_dir, 'cifar100_train.h5'), num_classes = 100, classes_shuff = classes_shuff)
        ts_set = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transforms.ToTensor())
        cifar_to_h5py(ts_set, os.path.join(save_dir, 'cifar100_test.h5'), isTrain = False, num_classes = 100, classes_shuff = classes_shuff)
    else:
        if not dset['link']: dset['link'] = 'https://github.com/gmshroff/metaLearning2022/raw/main/data/{}.zip'.format(dset['filename'])
        zip_file = download_dataset(dset['name'], dset['link'], save_dir)
        with zipfile.ZipFile(zip_file) as zf:
            zf.extractall(save_dir)
        os.remove(os.path.join(save_dir, f'{dset["name"]}.zip'))
        return os.path.join(save_dir, dset['filename'])

if __name__ == '__main__':

    datasets = {
        # 'image_data': {
        #     'name': 'image_data',
        #     'filename': 'image_data.pickle',
        #     'link' : '',
        # },
        # 'image_grid_data': {
        #     'name': 'image_grid_data',
        #     'filename': 'image_grid_data.pickle',
        #     'link' : '',
        # },
        'cifar100' : {
            'name': 'cifar100',
        }
    }

    for dset in datasets:
        prepare(datasets[dset])
