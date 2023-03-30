from collections import Counter
import io
import os
import pickle
import random

from boltons.iterutils import chunked
import lmdb
import numpy as np
from PIL import Image
import pysaliency
from pysaliency.datasets import create_subset
from pysaliency.utils import remove_trailing_nans
import torch
from tqdm import tqdm


def ensure_color_image(image):
    if len(image.shape) == 2:
        return np.dstack([image, image, image])
    return image


def x_y_to_sparse_indices(xs, ys):
    # Converts list of x and y coordinates into indices and values for sparse mask
    x_inds = []
    y_inds = []
    values = []
    pair_inds = {}

    for x, y in zip(xs, ys):
        key = (x, y)
        if key not in pair_inds:
            x_inds.append(x)
            y_inds.append(y)
            pair_inds[key] = len(x_inds) - 1
            values.append(1)
        else:
            values[pair_inds[key]] += 1

    return np.array([y_inds, x_inds]), values


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stimuli,
        fixations,
        centerbias_model=None,
        lmdb_path=None,
        transform=None,
        cached=None,
        average='fixation'
    ):
        self.stimuli = stimuli
        self.fixations = fixations
        self.centerbias_model = centerbias_model
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.average = average

        # cache only short dataset
        if cached is None:
            cached = len(self.stimuli) < 100

        cache_fixation_data = cached

        if lmdb_path is not None:
            _export_dataset_to_lmdb(stimuli, centerbias_model, lmdb_path)
            self.lmdb_env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                readonly=True, lock=False,
                readahead=False, meminit=False
            )
            cached = False
            cache_fixation_data = True
        else:
            self.lmdb_env = None

        self.cached = cached
        if cached:
            self._cache = {}
        self.cache_fixation_data = cache_fixation_data
        if cache_fixation_data:
            print("Populating fixations cache")
            self._xs_cache = {}
            self._ys_cache = {}

            for x, y, n in zip(self.fixations.x_int, self.fixations.y_int, tqdm(self.fixations.n)):
                self._xs_cache.setdefault(n, []).append(x)
                self._ys_cache.setdefault(n, []).append(y)

            for key in list(self._xs_cache):
                self._xs_cache[key] = np.array(self._xs_cache[key], dtype=int)
            for key in list(self._ys_cache):
                self._ys_cache[key] = np.array(self._ys_cache[key], dtype=int)

    def get_shapes(self):
        return list(self.stimuli.sizes)

    def _get_image_data(self, n):
        if self.lmdb_env:
            image, centerbias_prediction = _get_image_data_from_lmdb(self.lmdb_env, n)
        else:
            image = np.array(self.stimuli.stimuli[n])
            centerbias_prediction = self.centerbias_model.log_density(image)

            image = ensure_color_image(image).astype(np.float32)
            image = image.transpose(2, 0, 1)

        return image, centerbias_prediction

    def __getitem__(self, key):
        if not self.cached or key not in self._cache:

            image, centerbias_prediction = self._get_image_data(key)
            centerbias_prediction = centerbias_prediction.astype(np.float32)

            if self.cache_fixation_data and self.cached:
                xs = self._xs_cache.pop(key)
                ys = self._ys_cache.pop(key)
            elif self.cache_fixation_data and not self.cached:
                xs = self._xs_cache[key]
                ys = self._ys_cache[key]
            else:
                inds = self.fixations.n == key
                xs = np.array(self.fixations.x_int[inds], dtype=int)
                ys = np.array(self.fixations.y_int[inds], dtype=int)

            data = {
                "image": image,
                "x": xs,
                "y": ys,
                "centerbias": centerbias_prediction,
            }

            if self.average == 'image':
                data['weight'] = 1.0
            else:
                data['weight'] = float(len(xs))

            if self.cached:
                self._cache[key] = data
        else:
            data = self._cache[key]

        if self.transform is not None:
            return self.transform(dict(data))

        return data

    def __len__(self):
        return len(self.stimuli)


class FixationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stimuli, fixations,
        centerbias_model=None,
        lmdb_path=None,
        transform=None,
        included_fixations=-2,
        allow_missing_fixations=False,
        average='fixation',
        cache_image_data=False,
    ):
        self.stimuli = stimuli
        self.fixations = fixations
        self.centerbias_model = centerbias_model
        self.lmdb_path = lmdb_path

        if lmdb_path is not None:
            _export_dataset_to_lmdb(stimuli, centerbias_model, lmdb_path)
            self.lmdb_env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                readonly=True, lock=False,
                readahead=False, meminit=False
            )
            cache_image_data=False
        else:
            self.lmdb_env = None

        self.transform = transform
        self.average = average

        self._shapes = None

        if isinstance(included_fixations, int):
            if included_fixations < 0:
                included_fixations = [-1 - i for i in range(-included_fixations)]
            else:
                raise NotImplementedError()

        self.included_fixations = included_fixations
        self.allow_missing_fixations = allow_missing_fixations
        self.fixation_counts = Counter(fixations.n)

        self.cache_image_data = cache_image_data

        if self.cache_image_data:
            self.image_data_cache = {}

            print("Populating image cache")
            for n in tqdm(range(len(self.stimuli))):
                self.image_data_cache[n] = self._get_image_data(n)

    def get_shapes(self):
        if self._shapes is None:
            shapes = list(self.stimuli.sizes)
            self._shapes = [shapes[n] for n in self.fixations.n]

        return self._shapes

    def _get_image_data(self, n):
        if self.lmdb_path:
            return _get_image_data_from_lmdb(self.lmdb_env, n)
        image = np.array(self.stimuli.stimuli[n])
        centerbias_prediction = self.centerbias_model.log_density(image)

        image = ensure_color_image(image).astype(np.float32)
        image = image.transpose(2, 0, 1)

        return image, centerbias_prediction

    def __getitem__(self, key):
        n = self.fixations.n[key]

        if self.cache_image_data:
            image, centerbias_prediction = self.image_data_cache[n]
        else:
            image, centerbias_prediction = self._get_image_data(n)

        centerbias_prediction = centerbias_prediction.astype(np.float32)

        x_hist = remove_trailing_nans(self.fixations.x_hist[key])
        y_hist = remove_trailing_nans(self.fixations.y_hist[key])

        if self.allow_missing_fixations:
            _x_hist = []
            _y_hist = []
            for fixation_index in self.included_fixations:
                if fixation_index < -len(x_hist):
                    _x_hist.append(np.nan)
                    _y_hist.append(np.nan)
                else:
                    _x_hist.append(x_hist[fixation_index])
                    _y_hist.append(y_hist[fixation_index])
            x_hist = np.array(_x_hist)
            y_hist = np.array(_y_hist)
        else:
            print("Not missing")
            x_hist = x_hist[self.included_fixations]
            y_hist = y_hist[self.included_fixations]

        data = {
            "image": image,
            "x": np.array([self.fixations.x_int[key]], dtype=int),
            "y": np.array([self.fixations.y_int[key]], dtype=int),
            "x_hist": x_hist,
            "y_hist": y_hist,
            "centerbias": centerbias_prediction,
        }

        if self.average == 'image':
            data['weight'] = 1.0 / self.fixation_counts[n]
        else:
            data['weight'] = 1.0

        if self.transform is not None:
            return self.transform(data)

        return data

    def __len__(self):
        return len(self.fixations)


class FixationMaskTransform(object):
    def __init__(self, sparse=True):
        super().__init__()
        self.sparse = sparse

    def __call__(self, item):
        shape = torch.Size([item['image'].shape[1], item['image'].shape[2]])
        x = item.pop('x')
        y = item.pop('y')

        # inds, values = x_y_to_sparse_indices(x, y)
        inds = np.array([y, x])
        values = np.ones(len(y), dtype=int)

        mask = torch.sparse.IntTensor(torch.tensor(inds), torch.tensor(values), shape)
        mask = mask.coalesce()
        # sparse tensors don't work with workers...
        if not self.sparse:
            mask = mask.to_dense()

        item['fixation_mask'] = mask

        return item


class ImageDatasetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size=1, ratio_used=1.0, shuffle=True):
        self.ratio_used = ratio_used
        self.shuffle = shuffle

        shapes = data_source.get_shapes()
        unique_shapes = sorted(set(shapes))

        shape_indices = [[] for shape in unique_shapes]

        for k, shape in enumerate(shapes):
            shape_indices[unique_shapes.index(shape)].append(k)

        if self.shuffle:
            for indices in shape_indices:
                random.shuffle(indices)

        self.batches = sum([chunked(indices, size=batch_size) for indices in shape_indices], [])

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.batches))
        else:
            indices = range(len(self.batches))

        if self.ratio_used < 1.0:
            indices = indices[:int(self.ratio_used * len(indices))]

        return iter(self.batches[i] for i in indices)

    def __len__(self):
        return int(self.ratio_used * len(self.batches))


def _export_dataset_to_lmdb(stimuli: pysaliency.FileStimuli, centerbias_model: pysaliency.Model, lmdb_path, write_frequency=100):
    lmdb_path = os.path.expanduser(lmdb_path)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, stimulus in enumerate(tqdm(stimuli)):
        key = u'{}'.format(idx).encode('ascii')

        previous_data = txn.get(key)
        if previous_data:
            continue

        #timulus_data = stimulus.stimulus_data
        stimulus_filename = stimuli.filenames[idx]
        centerbias = centerbias_model.log_density(stimulus)

        txn.put(
            key,
            _encode_filestimulus_item(stimulus_filename, centerbias)
        )
        if idx % write_frequency == 0:
            #print("[%d/%d]" % (idx, len(stimuli)))
            #print("stimulus ids", len(stimuli.stimulus_ids._cache))
            #print("stimuli.cached", stimuli.cached)
            #print("stimuli", len(stimuli.stimuli._cache))
            #print("centerbias", len(centerbias_model._cache._cache))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    #keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    #with db.begin(write=True) as txn:
    #    txn.put(b'__keys__', dumps_pyarrow(keys))
    #    txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def _encode_filestimulus_item(filename, centerbias):
    with open(filename, 'rb') as f:
        image_bytes = f.read()

    buffer = io.BytesIO()
    pickle.dump({'image': image_bytes, 'centerbias': centerbias}, buffer)
    buffer.seek(0)
    return buffer.read()


def _get_image_data_from_lmdb(lmdb_env, n):
    key = '{}'.format(n).encode('ascii')
    with lmdb_env.begin(write=False) as txn:
        byteflow = txn.get(key)
    data = pickle.loads(byteflow)
    buffer = io.BytesIO(data['image'])
    buffer.seek(0)
    image = np.array(Image.open(buffer).convert('RGB'))
    centerbias_prediction = data['centerbias']
    image = image.transpose(2, 0, 1)

    return image, centerbias_prediction