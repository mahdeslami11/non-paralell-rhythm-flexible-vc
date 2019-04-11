import os
import sys
sys.path.insert(0,'..')
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

class VCTKDataset(Dataset):
    """VCTK dataset."""
    def __init__(self,
                 feat_dir,
                 meta_path,
                 dict_path):
        """Super class of VCTKDataset.

        Args:
            feat_dir: Directory for feat data.
            meta_path: Path to the meta data file.
            dict_path: Path to phone dictionary.
        """
        self.feat_dir = feat_dir
        self.meta_path = meta_path
        self.phone_dict = self._get_dict(dict_path)
        self.phone_dict_dim = len(self.phone_dict)

        # get feat paths
        self.feat_paths = self._get_path(feat_dir, meta_path)

    def __len__(self):
        return len(self.feat_paths)

    def _get_dict(self, dict_path):
        # sp: 0, sil: 1, let sp serve as <PAD> and <EOS>
        # <GO> will simply be zero vectors theni
        phone_dict = {}
        with open(dict_path) as f:
            for line in f:
                phn, num = line.strip().split(' ')
                phone_dict[phn] = num
        return phone_dict

    def _get_path(self, feat_dir, meta_path):
        feat_paths = []
        with open(meta_path) as f:
            for line in f:
                f_id = line.strip().split('|')[0]
                feat_path = os.path.join(feat_dir, '{}.pkl'.format(f_id))
                feat_paths.append(feat_path)
        return feat_paths

    def _my_pad(self, *args):
        result = []
        for arg in args:
            max_len = max([len(x) for x in arg])
            batch = [
                np.pad(x[0], [[0, max_len - len(x)], [0, 0]], 'constant')
                for x in arg
            ]
            result.append(batch)
        assert len(args) == len(result)
        return result

    def __getitem__(self):
        raise NotImplementedError()

    def _collate_fn(self):
        raise NotImplementedError()

class PPR_VCTKDataset(VCTKDataset):
    def __init__(self,
                 feat_dir,
                 meta_path,
                 dict_path):
        super(PPR_VCTKDataset, self).__init__(feat_dir, meta_path, dict_path)

    def __getitem__(self, index):
        with open(self.feat_paths[index], 'rb') as f:
            feat = pickle.load(f)
            mel, phn_seq = feat['mel'], feat['phn']
            label = [self.phone_dict[phn] for phn in phn_seq.strip().split(' ')]
        return mel, label

    def _collate_fn(self, batch):
        # batch: list of (feat_batch, label_batch) from __getitem__

        # Dynamic padding.
        feat_batch, label_batch = _my_pad(batch)
        feat_batch = torch.as_tensor(feat_batch)
        label_batch = torch.as_tensor(label_batch)

        return feat_batch, label_batch

class PPTS_VCTKDataset(VCTKDataset):
    def __init__(self,
                 feat_dir,
                 meta_path,
                 dict_path,
                 phn_hat_dir):
        super(PPTS_VCTKDataset, self).__init__(feat_dir, meta_path, dict_path)

        self.phn_hat_paths = [p.replace(feat_dir, phn_hat_dir) for p in self.feat_paths]

    def __getitem__(self, index):
        with open(self.phn_hat_paths[index], 'rb') as f, \
             open(self.feat_paths[index], 'rb') as g:
            phn_hat = pickle.load(f)
            feat = pickle.load(g)
            mag = feat['mag']
            assert len(phn_hat) == len(mag)
        return phn_hat, mag

    def _collate_fn(self, batch):
        # batch: list of (feat_batch, label_batch) from __getitem__

        # Dynamic padding.
        phn_batch, mag_batch = _my_pad(batch)
        phn_batch = torch.as_tensor(phn_batch)
        mag_batch = torch.as_tensor(mag_batch)

        return phn_batch, mag_batch

def UPPT_VCTKDataset(Dataset):
    def __init__(self):
        self.V = None
