import requests
from tqdm import tqdm
import os
import zipfile

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


def prepare(dset, save_dir = 'datasets'):
    if not dset['link']: dset['link'] = 'https://github.com/gmshroff/metaLearning2022/raw/main/data/{}.zip'.format(dset['filename'])
    zip_file = download_dataset(dset['name'], dset['link'], save_dir)
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(save_dir)
    os.remove(os.path.join(save_dir, f'{dset["name"]}.zip'))
    return os.path.join(save_dir, dset['filename'])

if __name__ == '__main__':

    datasets = {
        'image_data': {
            'name': 'image_data',
            'filename': 'image_data.pickle',
            'link' : '',
        },
        'image_grid_data': {
            'name': 'image_grid_data',
            'filename': 'image_grid_data.pickle',
            'link' : '',
        },
    }

    for dset in datasets:
        prepare(datasets[dset])
