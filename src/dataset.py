import os
import sys
sys.path.insert(0,'..')
import numpy as np
import pickle
from collections import defaultdict

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
                # only 20 speakers are used
                if f_id.split('_')[0][1:] in need:
                    feat_path = os.path.join(feat_dir, '{}.pkl'.format(f_id))
                    feat_paths.append(feat_path)
        return feat_paths

    def _pad_1d(self, _input, max_len):
        ''' pad 1d for a batch of input '''
        padded = [
            np.pad(x, (0, max_len - len(x)), 'constant')
            for x in _input
        ]
        return padded

    def _pad_2d(self, _input, max_len):
        ''' pad 2d for a batch of input '''
        padded = [
            np.pad(x, [[0, max_len - len(x)], [0, 0]], 'constant')
            for x in _input
        ]
        return padded

    def _pad_one_hot(self, _input, max_len, one_hot_dim, pad_idx):
        ''' pad one-hot vector for a batch of input '''
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

class UPPT_VCTKDataset(VCTKDataset):
    def __init__(self,
                 feat_dir,
                 meta_path,
                 dict_path,
                 phn_hat_dir,
                 A_id,
                 B_id,
                 max_len,
                 mode='train'):
        super(UPPT_VCTKDataset, self).__init__(feat_dir, meta_path, dict_path)
        self.A_id = A_id
        self.B_id = B_id
        self.max_len = max_len
        self.mode = mode
        self.phn_hat_paths = [
            p.replace(feat_dir, phn_hat_dir).replace('.pkl', '_phn_hat.pkl') \
                for p in self.feat_paths
        ]

        self.use_all_spk = True if self.A_id == 'all' and self.B_id == 'all' else False
        if self.use_all_spk:
            self.A_group = "p340 p231 p257 p363 p250 p285 p266 p361 p360".split(' ')
            self.B_group = "p256 p306 p301 p303 p376 p265 p268 p341 p251".split(' ')
            self._load_all_feat()
            self.data_nums = {'A':len(self.f_ids['A']), 'B': len(self.f_ids['B'])}
        else:
            self._load_feat()
            self.data_nums = {A_id:len(self.f_ids[A_id]), B_id: len(self.f_ids[B_id])}

    def _trim_sil(self, phn_hat):
        idx = 0
        while 1-phn_hat[idx][1] < 1e-5:
            idx += 1

        if idx < 10:
            idx = 0

        return phn_hat[idx:]

    def _load_feat(self):
        self.f_ids = defaultdict(list)
        self.phn_hats = defaultdict(list)
        self.original_len = defaultdict(dict)

        for fpath, ppath in zip(self.feat_paths, self.phn_hat_paths):
            cur_id = fpath.split('/')[-1].split('_')[0]
            if cur_id in [self.A_id, self.B_id]:
                with open(fpath, 'rb') as f, open(ppath, 'rb') as g:
                    phn_hat = np.array(pickle.load(g))

                    if len(phn_hat) > self.max_len:
                        continue
                    else:
                        phn_hat = self._trim_sil(phn_hat)
                        phn_hat = self._pad_one_hot(
                            np.expand_dims(phn_hat, 0),
                            self.max_len, phn_hat.shape[-1], 0
                        )[0]

                    feat = pickle.load(f)
                    self.f_ids[cur_id].append(feat['f_id'])
                    self.phn_hats[cur_id].append(phn_hat)
                    self.original_len[cur_id][feat['f_id']] = len(phn_hat)
        print("At {} mode, {}'s data loaded".format(self.mode, self.meta_path.split('/')[-1]))
        return

    def _load_all_feat(self):
        self.f_ids = defaultdict(list)
        self.phn_hats = defaultdict(list)
        self.original_len = defaultdict(dict)

        for fpath, ppath in zip(self.feat_paths, self.phn_hat_paths):
            cur_id = fpath.split('/')[-1].split('_')[0]
            if cur_id in self.A_group or cur_id in self.B_group:
                with open(fpath, 'rb') as f, open(ppath, 'rb') as g:
                    phn_hat = np.array(pickle.load(g))

                    if len(phn_hat) > self.max_len:
                        continue
                    else:
                        phn_hat = self._trim_sil(phn_hat)
                        phn_hat = self._pad_one_hot(
                            np.expand_dims(phn_hat, 0),
                            self.max_len, phn_hat.shape[-1], 0
                        )[0]

                    feat = pickle.load(f)

                    group = 'A' if cur_id in self.A_group else 'B'

                    self.f_ids[group].append(feat['f_id'])
                    self.phn_hats[group].append(phn_hat)
                    self.original_len[group][feat['f_id']] = len(phn_hat)
        print("At {} mode, {}'s data loaded".format(self.mode, self.meta_path.split('/')[-1]))
        return

    def __len__(self):
        if self.use_all_spk:
            return self.data_nums['A'] + self.data_nums['B']
        else:
            return self.data_nums[self.A_id] + self.data_nums[self.B_id]

    def __getitem__(self, index):
        # do not use index here for flexibility in sampling different speakers
        if self.use_all_spk:
            A_idx = np.random.randint(0, self.data_nums['A'])
            B_idx = np.random.randint(0, self.data_nums['B'])
            return self.f_ids['A'][A_idx], self.phn_hats['A'][A_idx], \
                   self.f_ids['B'][B_idx], self.phn_hats['B'][B_idx]
        else:
            A_idx = np.random.randint(0, self.data_nums[self.A_id])
            B_idx = np.random.randint(0, self.data_nums[self.B_id])
            return self.f_ids[self.A_id][A_idx], self.phn_hats[self.A_id][A_idx], \
                   self.f_ids[self.B_id][B_idx], self.phn_hats[self.B_id][B_idx]
    '''
    def _my_pad(self, batch):
        # f_ids
        f_ids = [[x[0] for x in batch]]

        # phn_distributions
        max_len_1 = max([len(x[1]) for x in batch])
        one_hot_dim = batch[0][1].shape[-1]
        phn_batch = self._pad_one_hot([x[1] for x in batch], max_len_1, one_hot_dim, 0)

        return f_ids, phn_batch
    '''
    def _collate_fn(self, batch):
        # batch: list of (f_id, phn_hat_batch, mag_batch) from __getitem__
        # pre-padding
        A_f_ids = [x[0] for x in batch]
        A_phn_batch = torch.as_tensor([x[1] for x in batch], dtype=torch.float)
        B_f_ids = [x[2] for x in batch]
        B_phn_batch = torch.as_tensor([x[3] for x in batch], dtype=torch.float)
        return A_f_ids, A_phn_batch, B_f_ids, B_phn_batch

        '''
        # Dynamic padding.
        A_f_ids, A_phn_batch = self._my_pad(batch[0:3])
        A_phn_batch = torch.as_tensor(A_phn_batch, dtype=torch.float)

        B_f_ids, B_phn_batch, B_mag_batch = self._my_pad(batch[3:])
        B_phn_batch = torch.as_tensor(B_phn_batch, dtype=torch.float)

        return A_f_ids, A_phn_batch, B_f_ids, B_phn_batch
        '''

class STAR_VCTKDataset(VCTKDataset):
    def __init__(self,
                 feat_dir,
                 meta_path,
                 dict_path,
                 phn_hat_dir,
                 max_len,
                 mode='train'):
        super(STAR_VCTKDataset, self).__init__(feat_dir, meta_path, dict_path)
        self.max_len = max_len
        self.mode = mode
        self.phn_hat_paths = [
            p.replace(feat_dir, phn_hat_dir).replace('.pkl', '_phn_hat.pkl') \
                for p in self.feat_paths
        ]
        '''
        self.A_group = "p340 p231 p257 p363 p250".split(' ')
        self.B_group = "p250 p285 p266 p361 p360".split(' ')
        self.C_group = "p256 p306 p301 p303 p376".split(' ')
        self.D_group = "p376 p265 p268 p341 p251".split(' ')
        '''
        self.A_group = "p340".split(' ')
        self.B_group = "p250".split(' ')
        self.C_group = "p256".split(' ')
        self.D_group = "p376".split(' ')
        self.all_ids = self.A_group + self.B_group + self.C_group + self.D_group
        self.class_num = 4
        self._build_id_map()

        self.data_num = self._load_feat()

    def _build_id_map(self):
        self.id_map = {}
        for ids in self.A_group:
            self.id_map[ids] = 0
        for ids in self.B_group:
            self.id_map[ids] = 1
        for ids in self.C_group:
            self.id_map[ids] = 2
        for ids in self.D_group:
            self.id_map[ids] = 3
        return

    def _trim_sil(self, phn_hat):
        idx = 0
        while 1-phn_hat[idx][1] < 1e-5:
            idx += 1

        if idx < 10:
            idx = 0

        return phn_hat[idx:]

    def _load_feat(self):
        self.f_ids = defaultdict(list)
        self.phn_hats = defaultdict(list)

        cnt = 0
        for fpath, ppath in zip(self.feat_paths, self.phn_hat_paths):
            cur_id = fpath.split('/')[-1].split('_')[0]
            if cur_id in (self.all_ids):
                with open(fpath, 'rb') as f, open(ppath, 'rb') as g:
                    phn_hat = np.array(pickle.load(g))

                    if len(phn_hat) > self.max_len:
                        continue
                    else:
                        phn_hat = self._trim_sil(phn_hat)
                        phn_hat = self._pad_one_hot(
                            np.expand_dims(phn_hat, 0),
                            self.max_len, phn_hat.shape[-1], 0
                        )[0]
                        cnt += 1

                    feat = pickle.load(f)
                    group = self.id_map[feat['f_id'].split('_')[0]]
                    
                    self.f_ids[group].append(feat['f_id'].split('.')[0])
                    self.phn_hats[group].append(phn_hat)
        print("At {} mode, {}'s data loaded".format(self.mode, self.meta_path.split('/')[-1]))
        return cnt

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        # do not use index here for flexibility in sampling different speakers
        rnd_list = [x for x in range(self.class_num)]
        src_group, tgt_group = rnd_list[0:2]
        src_idx = np.random.randint(0, len(self.phn_hats[src_group]))
        tgt_idx = np.random.randint(0, len(self.phn_hats[tgt_group]))
        return src_group, self.f_ids[src_group][src_idx], self.phn_hats[src_group][src_idx], \
               tgt_group, self.f_ids[tgt_group][tgt_idx], self.phn_hats[tgt_group][tgt_idx]

    def _collate_fn(self, batch):
        # batch: list of (group, f_id, phn_hat_batch) from __getitem__
        src_group = torch.as_tensor([x[0] for x in batch], dtype=torch.long)
        A_f_ids = [x[1] for x in batch]
        A_phn_batch = torch.as_tensor([x[2] for x in batch], dtype=torch.float)
        tgt_group = torch.as_tensor([x[3] for x in batch], dtype=torch.long)
        B_f_ids = [x[4] for x in batch]
        B_phn_batch = torch.as_tensor([x[5] for x in batch], dtype=torch.float)
        return src_group, A_f_ids, A_phn_batch, tgt_group, B_f_ids, B_phn_batch
