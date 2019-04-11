import torch

from models import PPR, PPTS, Generator, Discriminator
from dataset import PPR_VCTKDataset, PPTS_VCTKDataset, UPPT_VCTKDataset
from torch.utils.data import DataLoader

class Solver(object):
    def __init__(self, config, args, mode='train'):
        print(config['solver']['use_gpu'])
        self.use_gpu = config['solver']['use_gpu'] and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        self.config = config
        self.args = args
        self.mode = mode

        self.feat_dir = config['path']['feat_dir']
        self.meta_path = config['path']['{}_meta_path'.format(mode)]
        self.dict_path = config['path']['phn_dict_path']

        self.batch_size = config['solver']['batch_size']
        self.num_workers = config['solver']['num_workers']

class PPR_Solver(Solver):
    def __init__(self, config, args, mode='train'):
        super(PPR_Solver, self).__init__(config, args, mode)

        self.n_mels = config['audio']['n_mels']
        self.phn_dim = config['audio']['phn_dim']

    def get_dataset(self):
        dataset = PPR_VCTKDataset(
            feat_dir=self.feat_dir,
            meta_path=self.meta_path,
            dict_path=self.dict_path
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            collate_fn=dataset._collate_fn, pin_memory=True
        )
        return dataloader

    def build_model(self):
        ppr = PPR(
            input_dim=self.n_mels, output_dim=self.phn_dim, dropout_rate=0.5,
            prenet_hidden_dims=[256, 128], K=16,
            conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=128
        )
        print(ppr)
