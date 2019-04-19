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
        need = "340 231 257 363 250 285 266 361 360 256 306 301 303 376 265 268 341 251".split(' ')
        feat_paths = []
        with open(meta_path) as f:
            for line in f:
                f_id = line.strip().split('|')[0]
                # only SLT speakers are used
                if f_id.split('_')[0][1:] in need:
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

    def _pad_one_hot(self, _input, max_len, one_hot_dim, pad_idx):
        PAD_one_hot = np.zeros([1, one_hot_dim])
        PAD_one_hot[0][pad_idx] = 1
        padded = [
            np.concatenate(
                (x, PAD_one_hot.repeat(max_len - len(x), axis=0)),
                axis=0
            )
            for x in _input
        ]
        return padded

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self):
        raise NotImplementedError()

    def _load_feat(self):
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
        self.f_ids = []
        self.mels = []
        self.labels = []
        self.original_len = {}
        for path in self.feat_paths:
            with open(path, 'rb') as f:
                feat = pickle.load(f)
                mel, phn_seq = feat['mel'], feat['phn']
                # 1 for "sil", preventing OOV phoneme
                label = [self.phone_dict[phn] if phn in self.phone_dict else 1 \
                        for phn in phn_seq]
                self.f_ids.append(feat['f_id'])
                self.mels.append(np.array(mel))
                self.labels.append(np.array(label))
                self.original_len[feat['f_id']] = len(mel)
        print("At {} mode, {}'s data loaded".format(self.mode, self.meta_path.split('/')[-1]))
        return

    def __len__(self):
        return len(self.f_ids)

    def __getitem__(self, index):
        return self.f_ids[index], self.mels[index], self.labels[index]

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
    def _my_pad(self, batch):
        # f_ids
        f_ids = [[x[0] for x in batch]]

        # mels
        max_len_1 = max([len(x[1]) for x in batch])
        mel_batch = self._pad_2d([x[1] for x in batch], max_len_1)

        # labels
        max_len_2 = max([len(x[2]) for x in batch])
        label_batch = self._pad_1d([x[2] for x in batch], max_len_2)

        return f_ids, mel_batch, label_batch

    def _collate_fn(self, batch):
        # batch: list of (f_id, mel, label) from __getitem__

        # Dynamic padding.
        f_ids, mel_batch, label_batch = self._my_pad(batch)
        mel_batch = torch.as_tensor(mel_batch, dtype=torch.float)
        label_batch = torch.as_tensor(label_batch, dtype=torch.long)

        return f_ids, mel_batch, label_batch

class PPTS_VCTKDataset(VCTKDataset):
    def __init__(self,
                 feat_dir,
                 meta_path,
                 dict_path,
                 phn_hat_dir,
                 spk_id,
                 mode='train'):
        super(PPTS_VCTKDataset, self).__init__(feat_dir, meta_path, dict_path)
        self.spk_id = spk_id
        self.mode = mode
        self.phn_hat_paths = [
            p.replace(feat_dir, phn_hat_dir).replace('.pkl', '_phn_hat.pkl') \
                for p in self.feat_paths
        ]
        self._load_feat()

    def _load_feat(self):
        self.f_ids = []
        self.phn_hats = []
        self.mags = []
        self.labels = []
        self.original_len = {}
        for fpath, ppath in zip(self.feat_paths, self.phn_hat_paths):
            if fpath.split('/')[-1].split('_')[0] == self.spk_id:
                with open(fpath, 'rb') as f, open(ppath, 'rb') as g:
                    phn_hat = pickle.load(g)
                    feat = pickle.load(f)
                    mag = feat['mag']

                    self.f_ids.append(feat['f_id'])
                    self.phn_hats.append(np.array(phn_hat))
                    self.mags.append(np.array(mag))
                    self.original_len[feat['f_id']] = len(mag)
                    assert len(phn_hat) == len(mag)
        print("At {} mode, {}'s data loaded".format(self.mode, self.meta_path.split('/')[-1]))
        return

    def __len__(self):
        return len(self.f_ids)

    def __getitem__(self, index):
        return self.f_ids[index], self.phn_hats[index], self.mags[index]

    def _my_pad(self, batch):
        # f_ids
        f_ids = [[x[0] for x in batch]]

        # phn_distributions
        max_len_1 = max([len(x[1]) for x in batch])
        one_hot_dim = batch[0][1].shape[-1]
        phn_batch = self._pad_one_hot([x[1] for x in batch], max_len_1, one_hot_dim, 0)

        # mags
        max_len_2 = max([len(x[2]) for x in batch])
        mag_batch = self._pad_2d([x[2] for x in batch], max_len_2)

        return f_ids, phn_batch, mag_batch

    def _collate_fn(self, batch):
        # batch: list of (f_id, phn_hat_batch, mag_batch) from __getitem__

        # Dynamic padding.
        f_ids, phn_batch, mag_batch = self._my_pad(batch)
        phn_batch = torch.as_tensor(phn_batch, dtype=torch.float)
        mag_batch = torch.as_tensor(mag_batch, dtype=torch.float)

        return f_ids, phn_batch, mag_batch

def UPPT_VCTKDataset(Dataset):
    def __init__(self):
        self.V = None
