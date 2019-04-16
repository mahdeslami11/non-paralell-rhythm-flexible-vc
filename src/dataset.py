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
                phone_dict[phn] = int(num)
        return phone_dict

    def _get_path(self, feat_dir, meta_path):
        feat_paths = []
        with open(meta_path) as f:
            for line in f:
                f_id = line.strip().split('|')[0]
                feat_path = os.path.join(feat_dir, '{}.pkl'.format(f_id))
                feat_paths.append(feat_path)
        return feat_paths

    def _pad_1d(self, _input, max_len):
        padded = [
            np.pad(x, (0, max_len - len(x)), 'constant')
            for x in _input
        ]
        return padded

    def _pad_2d(self, _input, max_len):
        padded = [
            np.pad(x, [[0, max_len - len(x)], [0, 0]], 'constant')
            for x in _input
        ]
        return padded

    def _my_pad(self, batch):
        component_num = len(batch[0])
        padded_batch = []
        max_lens = [max([len(x[i]) for x in batch]) for i in range(component_num)]
        for idx in range(component_num):
            not_pad = [x[idx] for x in batch]
            dims = len(batch[0][idx].shape)
            if dims == 1:
                padded = self._pad_1d(not_pad, max_lens[idx])
            elif dims == 2:
                padded = self._pad_2d(not_pad, max_lens[idx])
            else:
                raise NotImplementedError()
            padded_batch.append(padded)
        assert len(batch) == len(padded_batch[0])
        return padded_batch

    def __getitem__(self):
        raise NotImplementedError()

    def _collate_fn(self):
        raise NotImplementedError()

class PPR_VCTKDataset(VCTKDataset):
    def __init__(self,
                 feat_dir,
                 meta_path,
                 dict_path,
                 mode='train'):
        super(PPR_VCTKDataset, self).__init__(feat_dir, meta_path, dict_path)
        self.mode = mode
        self._load_feat()

    def _load_feat(self):
        need = "340 231 257 363 250 285 266 361 360 256 306 301 303 376 265 268 341 251".split(' ')

        self.f_ids = []
        self.mels = []
        self.labels = []
        for path in self.feat_paths:
            with open(path, 'rb') as f:
                feat = pickle.load(f)
                mel, phn_seq = feat['mel'], feat['phn']
                # 1 for "sil", preventing OOV
                label = [self.phone_dict[phn] if phn in self.phone_dict else 1 \
                        for phn in phn_seq]
                if feat['f_id'].split('_')[0][1:] in need:
                    self.f_ids.append(feat['f_id'])
                    self.mels.append(np.array(mel))
                    self.labels.append(np.array(label))
        return

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.mels[index], self.labels[index]
        elif self.mode == 'test':
            return self.f_id[index], self.mels[index], self.labels[index]
        else:
            raise NotImplementedError()

    '''
    def __getitem__(self, index):
        with open(self.feat_paths[index], 'rb') as f:
            feat = pickle.load(f)
            mel, phn_seq = feat['mel'], feat['phn']
            label = [self.phone_dict[phn] for phn in phn_seq]

        if self.mode == 'train':
            return np.array(mel), np.array(label)
        elif self.mode == 'test':
            return feat['f_id'], np.array(mel), np.array(label)
        else:
            raise NotImplementedError()
    '''
    def _collate_fn(self, batch):
        # batch: list of (mel, label) or (f_id, mel, label) from __getitem__

        batch = batch if self.mode == 'train' else batch[1:]
        # Dynamic padding.
        mel_batch, label_batch = self._my_pad(batch)
        mel_batch = torch.as_tensor(mel_batch, dtype=torch.float)
        label_batch = torch.as_tensor(label_batch, dtype=torch.long)

        if self.mode == 'train':
            return mel_batch, label_batch
        elif self.mode == 'test':
            f_ids = [x[0] for x in batch]
            return f_ids, mel_batch, label_batch
        else:
            raise NotImplementedError()

class PPTS_VCTKDataset(VCTKDataset):
    def __init__(self,
                 feat_dir,
                 meta_path,
                 dict_path,
                 phn_hat_dir,
                 mode='train'):
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
        phn_batch, mag_batch = self._my_pad(batch)
        phn_batch = torch.as_tensor(phn_batch)
        mag_batch = torch.as_tensor(mag_batch)

        return phn_batch, mag_batch

def UPPT_VCTKDataset(Dataset):
    def __init__(self):
        self.V = None
