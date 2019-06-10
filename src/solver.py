import os
import itertools
import pickle
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from .models import PPR, PPTS, Generator, Discriminator, OnehotEncoder
from .dataset import PPR_VCTKDataset, PPTS_VCTKDataset, UPPT_VCTKDataset
from .utils import AudioProcessor

class Solver(object):
    def __init__(self, config, args):
        self.use_gpu = config['solver']['use_gpu'] and torch.cuda.is_available()
        if self.use_gpu:
            print('Using {} GPUs!'.format(torch.cuda.device_count()))
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        self.config = config
        self.args = args

        self.ap = AudioProcessor(**config['audio'])

        self.feat_dir = config['path']['feat_dir']
        self.train_meta_path = config['path']['train_meta_path']
        self.eval_meta_path = config['path']['eval_meta_path']
        self.test_meta_path = config['path']['test_meta_path']
        self.dict_path = config['path']['phn_dict_path']

        self.batch_size = config['solver']['batch_size']
        self.num_workers = config['solver']['num_workers']
        self.log_interval = config['solver']['log_interval']
        self.summ_interval = config['solver']['summ_interval']
        self.ckpt_interval = config['solver']['ckpt_interval']

        # must-implement functions for Solver
        def get_dataset(self):
            raise NotImplementedError()

        def build_model(self):
            raise NotImplementedError()

        def save_ckpt(self):
            raise NotImplementedError()

        def load_ckpt(self):
            raise NotImplementedError()

        def train(self):
            raise NotImplementedError()

        def eval(self):
            raise NotImplementedError()

        def test(self):
            raise NotImplementedError()

class PPR_Solver(Solver):
    def __init__(self, config, args, mode='train'):
        super(PPR_Solver, self).__init__(config, args)

        self.n_mels = config['audio']['n_mels']
        self.phn_dim = config['text']['phn_dim']

        self.lr = config['model']['ppr']['lr']
        self.optimizer_type = config['model']['ppr']['type']
        self.betas = [float(x) for x in config['model']['ppr']['betas'].split(',')]
        self.weight_decay = config['model']['ppr']['weight_decay']
        self.label_smoothing = config['model']['ppr']['label_smoothing']

        self.mode = mode
        self.model = self.build_model()
        if mode == 'train':
            self.train_loader, _ = self.get_dataset(self.train_meta_path)
            self.eval_loader, _ = self.get_dataset(self.eval_meta_path)
            self.optimizer = self.build_optimizer()
            self.log_dir = config['path']['ppr']['log_dir']
            self.writer = SummaryWriter(self.log_dir)
            self.one_hot_encoder = OnehotEncoder(self.phn_dim)
        elif mode == 'test':
            self.test_loader, self.test_ori_len = self.get_dataset(self.test_meta_path)
            # also need to infer train and eval part for training UPPT and PPTS
            self.train_loader, self.train_ori_len = self.get_dataset(self.train_meta_path)
            self.eval_loader, self.eval_ori_len = self.get_dataset(self.eval_meta_path)

        self.save_dir = config['path']['ppr']['save_dir']
        self.ppr_output_dir = config['path']['ppr']['output_dir']

        # attempt to load or set gs and epoch to 0
        self.load_ckpt()

    def get_dataset(self, meta_path):
        dataset = PPR_VCTKDataset(
            feat_dir=self.feat_dir,
            meta_path=meta_path,
            dict_path=self.dict_path,
            mode=self.mode
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True if self.mode != 'test' else False,
            num_workers=self.num_workers,
            collate_fn=dataset._collate_fn, pin_memory=True
        )
        return dataloader, dataset.original_len

    def build_model(self):
        ppr = PPR(
            input_dim=self.n_mels, output_dim=self.phn_dim, dropout_rate=0.5,
            prenet_hidden_dims=[256, 128], K=16,
            conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=128
        )
        ppr = ppr.to(self.device)
        return ppr

    def build_optimizer(self):
        optimizer = getattr(torch.optim, self.optimizer_type)
        optimizer = optimizer(
            self.model.parameters(), lr=self.lr,
            betas=self.betas, weight_decay=self.weight_decay
        )
        return optimizer

    def save_ckpt(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        checkpoint_path = os.path.join(
            self.save_dir, "model.ckpt-{}.pt".format(self.global_step)
        )
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch
        }, checkpoint_path)
        print("Checkpoint model.ckpt-{}.pt saved.".format(self.global_step))
        with open(os.path.join(self.save_dir, "checkpoint"), "w") as f:
            f.write("model.ckpt-{}".format(self.global_step))
        return

    def load_ckpt(self):
        checkpoint_list = os.path.join(self.save_dir, 'checkpoint')
        if os.path.exists(checkpoint_list):
            checkpoint_filename = open(checkpoint_list).readline().strip()
            checkpoint_path = os.path.join(self.save_dir, "{}.pt".format(checkpoint_filename))
            if self.use_gpu:
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model'])
            if self.mode == 'train':
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']
            print("Checkpoint model.ckpt-{}.pt loaded.".format(self.global_step))
        else:
            self.global_step = 0
            self.epoch = 0
            print("Start training with new parameters.")
        return

    def _label_smoothing(self, label_hat, label_batch):
        # label_hat: log-likelihood, label_batch: list of int
        eps = 0.05
        one_hot = self.one_hot_encoder(label_batch)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (label_hat.shape[-1] - 1)
        one_hot = one_hot.to(self.device)
        loss = -(one_hot * label_hat).sum(-1).mean()
        return loss

    def _calc_acc(self, label_hat, label_batch):
        if isinstance(label_hat, torch.Tensor):
            label_hat = label_hat.detach().cpu().numpy()
        if isinstance(label_batch, torch.Tensor):
            label_batch = label_batch.detach().cpu().numpy()
        pred = np.argmax(label_hat, axis=2)
        acc = np.mean(pred == label_batch)
        return acc

    def train(self):
        epoch_loss = 0.0
        self.model.train()
        for idx, (_, mel_batch, label_batch) in enumerate(self.train_loader):
            mel_batch, label_batch = mel_batch.to(self.device), label_batch.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            label_hat = self.model(mel_batch)
            if self.label_smoothing:
                loss = self._label_smoothing(label_hat, label_batch)
            else:
                loss = F.nll_loss(torch.transpose(label_hat, 1, 2), label_batch)
            epoch_loss += loss.item()

            # Logging
            if self.global_step % self.log_interval == 0:
                print(
                    '[GS=%3d, epoch=%d, idx=%3d] loss: %.6f' % \
                    (self.global_step+1, self.epoch, idx+1, loss.item())
                )
            if self.global_step % self.summ_interval == 0:
                self.writer.add_scalar('train/training_loss', loss.item(), self.global_step)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Saving or not
            self.global_step += 1
            if self.global_step % self.ckpt_interval == 0:
                self.save_ckpt()

        epoch_loss /= (idx+1)
        print('[epoch %d] training_loss: %.6f' % (self.epoch, epoch_loss))
        self.writer.add_scalar('train/epoch_training_loss', epoch_loss, self.epoch)
        self.writer.add_image(
            'train/label_gt', torch.t(self.one_hot_encoder(label_batch[0])).detach().cpu().numpy(),
            self.epoch, dataformats='HW'
        )
        self.writer.add_image(
            'train/label_hat', torch.t(torch.exp(label_hat[0])).detach().cpu().numpy(),
            self.epoch, dataformats='HW'
        )
        self.epoch += 1

        return

    def eval(self):
        eval_loss = 0.0
        avg_acc = 0.0
        self.model.eval()
        with torch.no_grad():
            for idx, (_, mel_batch, label_batch) in enumerate(self.eval_loader):
                mel_batch, label_batch = mel_batch.to(self.device), label_batch.to(self.device)
                label_hat = self.model(mel_batch)
                loss = F.nll_loss(torch.transpose(label_hat, 1, 2), label_batch)
                eval_loss += loss.item()
                acc = self._calc_acc(label_hat, label_batch)
                avg_acc += acc

                if idx % 100 == 0:
                    break

        eval_loss /= (idx+1)
        avg_acc /= (idx+1)
        print('[eval %d] eval_loss: %.6f' % (self.epoch, eval_loss))
        print('[eval %d] eval_acc: %.6f' % (self.epoch, avg_acc))

        self.writer.add_scalar('eval/eval_loss', eval_loss, self.epoch)
        self.writer.add_scalar('eval/accuracy', avg_acc, self.epoch)
        self.writer.add_image(
            'eval/label_gt', torch.t(self.one_hot_encoder(label_batch[0])).detach().cpu().numpy(),
            self.epoch, dataformats='HW'
        )
        self.writer.add_image('eval/label_hat',
            torch.t(torch.exp(label_hat[0])).detach().cpu().numpy(),
            self.epoch, dataformats='HW'
        )

        return

    def save_label_hat(self, f_id, l_hat):
        out_name = "{}_phn_hat.pkl".format(f_id)
        out_path = os.path.join(self.ppr_output_dir, out_name)
        with open(out_path, 'wb') as f:
            pickle.dump(l_hat, f)
        return

    def test(self):
        if not os.path.exists(self.ppr_output_dir):
            os.makedirs(self.ppr_output_dir)
        self.model.eval()
        with torch.no_grad():
            # also need to infer train and eval part for training UPPT and PPTS
            for ori_len, loader in zip([self.test_ori_len, self.train_ori_len, self.eval_ori_len], \
                    [self.test_loader, self.train_loader, self.eval_loader]):
                for (f_id_list, mel_batch, label_batch) in loader:
                    mel_batch, label_batch = mel_batch.to(self.device), label_batch.to(self.device)
                    label_hat = torch.exp(self.model(mel_batch)).detach().cpu().numpy()
                    for f_id, l_hat in zip(f_id_list, label_hat):
                        self.save_label_hat(f_id, l_hat[:ori_len[f_id]])
        return

class PPTS_Solver(Solver):
    def __init__(self, config, args, mode='train'):
        super(PPTS_Solver, self).__init__(config, args)

        self.phn_hat_dir = config['path']['ppr']['output_dir']
        self.phn_dim = config['text']['phn_dim']
        self.n_fft = config['audio']['n_fft']

        self.lr = config['model']['ppts']['lr']
        self.optimizer_type = config['model']['ppts']['type']
        self.betas = [float(x) for x in config['model']['ppts']['betas'].split(',')]
        self.weight_decay = config['model']['ppts']['weight_decay']

        self.spk_id = args.spk_id
        if self.spk_id == ('' or None):
            print("[Error] A spk_id must be given to init a PPTS solver")
            exit()

        self.mode = mode
        self.model, self.criterion = self.build_model()
        if mode == 'train':
            self.optimizer = self.build_optimizer()
            self.train_loader = self.get_dataset(self.train_meta_path)
            self.eval_loader = self.get_dataset(self.eval_meta_path)
            self.log_dir = os.path.join(config['path']['ppts']['log_dir'], self.spk_id)
            self.writer = SummaryWriter(self.log_dir)
        elif mode == 'test':
            self.test_loader = self.get_dataset(self.test_meta_path)

        self.save_dir = os.path.join(config['path']['ppts']['save_dir'], self.spk_id)
        self.ppts_output_dir = os.path.join(config['path']['ppts']['output_dir'], self.spk_id)

        # attempt to load or set gs and epoch to 0
        self.load_ckpt()

    def get_dataset(self, meta_path):
        dataset = PPTS_VCTKDataset(
            feat_dir=self.feat_dir,
            meta_path=meta_path,
            dict_path=self.dict_path,
            phn_hat_dir=self.phn_hat_dir,
            spk_id = self.spk_id,
            mode=self.mode
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True if self.mode == 'train' else False,
            num_workers=self.num_workers,
            collate_fn=dataset._collate_fn, pin_memory=True
        )
        return dataloader

    def build_model(self):
        ppts = PPTS(
            input_dim=self.phn_dim, output_dim=(self.n_fft//2)+1,
            dropout_rate=0.5, prenet_hidden_dims=[256, 128], K=16,
            conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=256,
            gru_dim=256
        )
        ppts = ppts.to(self.device)
        criterion = torch.nn.L1Loss()
        return ppts, criterion

    def build_optimizer(self):
        optimizer = getattr(torch.optim, self.optimizer_type)
        optimizer = optimizer(
            self.model.parameters(), lr=self.lr,
            betas=self.betas, weight_decay=self.weight_decay
        )
        return optimizer

    def save_ckpt(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        checkpoint_path = os.path.join(
            self.save_dir, "model.ckpt-{}.pt".format(self.global_step)
        )
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch
        }, checkpoint_path)
        print("Checkpoint model.ckpt-{}.pt saved.".format(self.global_step))
        with open(os.path.join(self.save_dir, "checkpoint"), "w") as f:
            f.write("model.ckpt-{}".format(self.global_step))
        return

    def load_ckpt(self):
        checkpoint_list = os.path.join(self.save_dir, 'checkpoint')
        if os.path.exists(checkpoint_list):
            checkpoint_filename = open(checkpoint_list).readline().strip()
            checkpoint_path = os.path.join(self.save_dir, "{}.pt".format(checkpoint_filename))
            if self.use_gpu:
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model'])
            if self.mode == 'train':
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']
            print("Checkpoint model.ckpt-{}.pt loaded.".format(self.global_step))
        else:
            self.global_step = 0
            self.epoch = 0
            print("Start training with new parameters.")
        return

    def train(self):
        epoch_loss = 0.0
        self.model.train()
        for idx, (_, phn_hat_batch, mag_batch) in enumerate(self.train_loader):
            phn_hat_batch, mag_batch = phn_hat_batch.to(self.device), mag_batch.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            mag_hat = self.model(phn_hat_batch)
            #loss = self.criterion(mag_hat, mag_batch)
            loss = 0.5 * self.criterion(mag_hat, mag_batch) + \
                   0.5 * self.criterion(mag_hat[:,:,:200], mag_batch[:,:,:200])
            epoch_loss += loss.item()

            # Logging
            # Because of number of batch is too few, only log at epoch level
            '''
            if self.global_step % self.log_interval == 0:
                print(
                    '[GS=%3d, epoch=%d, idx=%3d] loss: %.6f' % \
                    (self.global_step+1, self.epoch+1, idx+1, loss.item())
                )
            if self.global_step % self.summ_interval == 0:
                self.writer.add_scalar('train/training_loss', loss.item(), self.global_step)
            '''

            # Backward
            loss.backward()
            self.optimizer.step()

            # Saving or not
            self.global_step += 1
            if self.global_step % self.ckpt_interval == 0:
                self.save_ckpt()

        epoch_loss /= (idx+1)
        print('[epoch %d] training_loss: %.6f' % (self.epoch, epoch_loss))
        self.writer.add_scalar('train/epoch_training_loss', epoch_loss, self.epoch)
        self.writer.add_image('train/phn_hat',
            torch.t(phn_hat_batch[0]).detach().cpu().numpy(),
            self.epoch, dataformats='HW'
        )
        self.writer.add_image(
            'train/mag_gt', torch.t(mag_batch[0]).detach().cpu().numpy()[::-1,:],
            self.epoch, dataformats='HW'
        )
        self.writer.add_image(
            'train/mag_hat', torch.t(mag_hat[0]).detach().cpu().numpy()[::-1,:],
            self.epoch, dataformats='HW'
        )
        self.writer.add_audio(
            'train/audio_gt', self.ap.inv_spectrogram(mag_batch[0].detach().cpu().numpy()),
            self.epoch, sample_rate=self.ap.sr
        )
        self.writer.add_audio(
            'train/audio_hat', self.ap.inv_spectrogram(mag_hat[0].detach().cpu().numpy()),
            self.epoch, sample_rate=self.ap.sr
        )
        self.epoch += 1

        return

    def eval(self):
        eval_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for idx, (_, phn_hat_batch, mag_batch) in enumerate(self.eval_loader):
                phn_hat_batch, mag_batch = phn_hat_batch.to(self.device), mag_batch.to(self.device)
                mag_hat = self.model(phn_hat_batch)
                loss = self.criterion(mag_hat, mag_batch)
                eval_loss += loss.item()

                if idx % 100 == 0:
                    break

        eval_loss /= (idx+1)
        print('[eval %d] eval_loss: %.6f' % (self.epoch, eval_loss))

        self.writer.add_scalar('eval/eval_loss', eval_loss, self.epoch)
        self.writer.add_image(
            'eval/mag_gt', torch.t(mag_batch[0]).detach().cpu().numpy()[::-1,:],
            self.epoch, dataformats='HW'
        )
        self.writer.add_image(
            'eval/mag_hat', torch.t(mag_hat[0]).detach().cpu().numpy()[::-1,:],
            self.epoch, dataformats='HW'
        )
        self.writer.add_audio(
            'eval/audio_gt', self.ap.inv_spectrogram(mag_batch[0].detach().cpu().numpy()),
            self.epoch, sample_rate=self.ap.sr
        )
        self.writer.add_audio(
            'eval/audio_hat', self.ap.inv_spectrogram(mag_hat[0].detach().cpu().numpy()),
            self.epoch, sample_rate=self.ap.sr
        )

        return

class UPPT_Solver(Solver):
    def __init__(self, config, args, mode='train'):
        super(UPPT_Solver, self).__init__(config, args)

        self.phn_hat_dir = config['path']['ppr']['output_dir']
        self.phn_dim = config['text']['phn_dim']

        self.lr = config['model']['uppt']['lr']
        self.optimizer_type = config['model']['uppt']['type']
        self.betas = [float(x) for x in config['model']['uppt']['betas'].split(',')]
        self.weight_decay = config['model']['uppt']['weight_decay']
        self.max_len = config['model']['uppt']['max_len']

        self.A_id = args.A_id
        self.B_id = args.B_id
        if self.A_id == ('' or None) or self.B_id == ('' or None):
            print("[Error] A speaker ID pair must be given to init a UPPT solver")
            exit()
        self.save_prefix = "from_{}_to_{}".format(self.A_id, self.B_id)

        self.mode = mode
        self.gen_A_to_B, self.gen_B_to_A = self.build_gen()
        if mode == 'train':
            self.dis_A, self.dis_B = self.build_dis()
            self.gen_optimizer, self.dis_optimizer = self.build_optimizer()
            self.train_loader = self.get_dataset(self.train_meta_path)
            self.eval_loader = self.get_dataset(self.eval_meta_path)
            self.log_dir = os.path.join(config['path']['uppt']['log_dir'], self.save_prefix)
            self.writer = SummaryWriter(self.log_dir)
            self.pre_train = args.pre_train
        elif mode == 'test':
            self.test_loader = self.get_dataset(self.test_meta_path)

        self.save_dir = os.path.join(config['path']['uppt']['save_dir'], self.save_prefix)
        self.uppt_output_dir = os.path.join(config['path']['uppt']['output_dir'], self.save_prefix)

        # attempt to load or set gs and epoch to 0
        self.load_ckpt()

    def get_dataset(self, meta_path):
        dataset = UPPT_VCTKDataset(
            feat_dir=self.feat_dir,
            meta_path=meta_path,
            dict_path=self.dict_path,
            phn_hat_dir=self.phn_hat_dir,
            A_id=self.A_id,
            B_id=self.B_id,
            max_len=self.max_len,
            mode=self.mode
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True if self.mode == 'train' else False,
            num_workers=self.num_workers,
            collate_fn=dataset._collate_fn, pin_memory=True
        )
        return dataloader

    def build_gen(self):
        gen_A_to_B = Generator(
            input_dim=self.phn_dim, r=3, dropout_rate=0.5,
            prenet_hidden_dims=[256,128], K=16,
            conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=128,
            max_decode_len=self.max_len
        ).to(self.device)

        gen_B_to_A = Generator(
            input_dim=self.phn_dim, r=3, dropout_rate=0.5,
            prenet_hidden_dims=[256,128], K=16,
            conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=128,
            max_decode_len=self.max_len
        ).to(self.device)

        return gen_A_to_B, gen_B_to_A

    def build_dis(self):
        dis_A = Discriminator(input_dim=self.phn_dim, input_len=self.max_len).to(self.device)
        dis_B = Discriminator(input_dim=self.phn_dim, input_len=self.max_len).to(self.device)

        return dis_A, dis_B

    def build_optimizer(self):
        optimizer = getattr(torch.optim, self.optimizer_type)
        gen_optimizer = optimizer(
            itertools.chain(self.gen_A_to_B.parameters(), self.gen_B_to_A.parameters()),
            lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        dis_optimizer = optimizer(
            itertools.chain(self.dis_A.parameters(), self.dis_B.parameters()),
            lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )

        return gen_optimizer, dis_optimizer

    def save_ckpt(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        checkpoint_path = os.path.join(
            self.save_dir, "model.ckpt-{}.pt".format(self.global_step)
        )
        torch.save({
            "gen_A_to_B": self.gen_A_to_B.state_dict(),
            "gen_B_to_A": self.gen_B_to_A.state_dict(),
            "dis_A": self.dis_A.state_dict(),
            "dis_B": self.dis_B.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict(),
            "dis_optimizer": self.dis_optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch
        }, checkpoint_path)
        print("Checkpoint model.ckpt-{}.pt saved.".format(self.global_step))
        with open(os.path.join(self.save_dir, "checkpoint"), "w") as f:
            f.write("model.ckpt-{}".format(self.global_step))
        return

    def load_ckpt(self):
        checkpoint_list = os.path.join(self.save_dir, 'checkpoint')
        if os.path.exists(checkpoint_list):
            checkpoint_filename = open(checkpoint_list).readline().strip()
            checkpoint_path = os.path.join(self.save_dir, "{}.pt".format(checkpoint_filename))
            if self.use_gpu:
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.gen_A_to_B.load_state_dict(checkpoint['gen_A_to_B'])
            self.gen_B_to_A.load_state_dict(checkpoint['gen_B_to_A'])
            if self.mode == 'train':
                self.dis_A.load_state_dict(checkpoint['dis_A'])
                self.dis_B.load_state_dict(checkpoint['dis_B'])
                self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
                self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']
            print("Checkpoint model.ckpt-{}.pt loaded.".format(self.global_step))
        else:
            self.global_step = 0
            self.epoch = 0
            print("Start training with new parameters.")
        return

    def _xent_loss(self, output, target):
        '''
        performing pure cross entropy calculation:
        xent(output, target) = -p(target) * log(q(output))
        '''
        return (-1 * target * output.log()).mean()

    def AE_step(self, mode):
        tf_rate = max(0.5, 1-(self.epoch)*0.0005) if mode == 'train' else 0.

        if mode == 'train':
            self.gen_optimizer.zero_grad()
        self.A_to_B, self.AB_attn = self.gen_A_to_B(self.A_batch, teacher_force_rate=tf_rate)
        self.B_to_A, self.BA_attn = self.gen_B_to_A(self.B_batch, teacher_force_rate=tf_rate)
        self.loss_AE_A = self._xent_loss(self.A_to_B, self.A_batch)
        self.loss_AE_B = self._xent_loss(self.B_to_A, self.B_batch)
        self.loss_AE = self.loss_AE_A + self.loss_AE_B
        if mode == 'train':
            self.loss_AE.backward()
            self.gen_optimizer.step()

        return

    def G_step(self, mode):
        tf_rate = 0.

        ##### Generators #####
        if mode == 'train':
            self.gen_optimizer.zero_grad()

        # GAN Loss
        self.A_to_B, self.AB_attn = self.gen_A_to_B(self.A_batch, teacher_force_rate=tf_rate)
        self.v_B_fake = self.dis_B(self.A_to_B)
        self.loss_GAN_A_to_B = -1*(self.v_B_fake).mean()

        self.B_to_A, self.BA_attn = self.gen_B_to_A(self.B_batch, teacher_force_rate=tf_rate)
        self.v_A_fake = self.dis_A(self.B_to_A)
        self.loss_GAN_B_to_A = -1*(self.v_A_fake).mean()

        # Cycle-consistency Loss
        self.A_to_B_to_A, _ = self.gen_B_to_A(self.A_to_B, teacher_force_rate=tf_rate)
        self.B_to_A_to_B, _ = self.gen_A_to_B(self.B_to_A, teacher_force_rate=tf_rate)
        self.loss_cycle_ABA = self._xent_loss(self.A_to_B_to_A, self.A_batch)
        self.loss_cycle_BAB = self._xent_loss(self.B_to_A_to_B, self.B_batch)

        # Identity Loss
        self.A_to_A, _ = self.gen_B_to_A(self.A_batch, teacher_force_rate=tf_rate)
        self.B_to_B, _ = self.gen_A_to_B(self.B_batch, teacher_force_rate=tf_rate)
        self.loss_identity_A = self._xent_loss(self.A_to_A, self.A_batch)
        self.loss_identity_B = self._xent_loss(self.B_to_B, self.B_batch)

        # Total Loss
        self.loss_G = self.loss_GAN_A_to_B + self.loss_GAN_B_to_A + \
                 self.loss_cycle_ABA*10 + self.loss_cycle_BAB*10 + \
                 self.loss_identity_A*5 + self.loss_identity_B*5
        if mode == 'train':
            self.loss_G.backward()
            self.gen_optimizer.step()

        return

    def D_step(self, mode):
        ##### Discriminators #####
        if mode == 'train':
            self.dis_optimizer.zero_grad()

        # GAN Loss (W distance)
        self.v_A_real = self.dis_A(self.A_batch)
        self.v_B_real = self.dis_B(self.B_batch)
        self.v_A_fake = self.dis_A(self.B_to_A.detach())
        self.v_B_fake = self.dis_B(self.A_to_B.detach())
        self.W_A = (self.v_A_real - self.v_A_fake).mean()
        self.W_B = (self.v_B_real - self.v_B_fake).mean()

        if mode == 'train':
            # Gradient Penalty
            cur_batch_size = self.A_batch.shape[0]
            alpha = torch.rand(cur_batch_size, 1, 1).to(self.device)
            self.A_inter = alpha * self.B_to_A.detach() + (1.0-alpha) * self.A_batch
            self.A_inter = Variable(self.A_inter, requires_grad=True).to(self.device)
            self.v_A_inter = self.dis_A(self.A_inter)
            gradient_A = torch.autograd.grad(
                self.v_A_inter, self.A_inter,
                grad_outputs=torch.ones(self.v_A_inter.size()).to(self.device),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            self.GP_A = ((gradient_A.norm(2) -1) ** 2).mean()

            alpha = torch.rand(cur_batch_size, 1, 1).to(self.device)
            self.B_inter = alpha * self.A_to_B.detach() + (1.0-alpha) * self.B_batch
            self.B_inter = Variable(self.B_inter, requires_grad=True).to(self.device)
            self.v_B_inter = self.dis_B(self.B_inter)
            gradient_B = torch.autograd.grad(
                self.v_B_inter, self.B_inter,
                grad_outputs=torch.ones(self.v_B_inter.size()).to(self.device),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            self.GP_B = ((gradient_B.norm(2) -1) ** 2).mean()

            # Total Loss
            self.loss_D = -1*(self.W_A + self.W_B) + 10*(self.GP_A + self.GP_B)
            self.loss_D.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                itertools.chain(self.dis_A.parameters(), self.dis_B.parameters()), 5.
            )
            self.dis_optimizer.step()

        else:
            # for recording W distance only
            self.loss_D = -1*(self.W_A + self.W_B)

        return

    def _add_image(self, name, data):
        self.writer.add_image(name,
            torch.t(data).detach().cpu().numpy(),
            self.epoch, dataformats='HW'
        )
        return

    def AE_logging(self, mode, batch_num):
        # Recording avg. loss in a epoch
        if mode == 'train':
            print('[epoch %d] training_loss: %.6f' % (self.epoch, self.epoch_loss/batch_num))
            self.writer.add_scalar(
                'train/AE/epoch_training_loss',
                self.epoch_loss/batch_num,
                self.global_step
            )
        else:
            print('[epoch %d] eval_loss: %.6f' % (self.epoch, self.eval_loss/batch_num))
            self.writer.add_scalar(
                'eval/AE/epoch_training_loss',
                self.eval_loss/batch_num,
                self.global_step
            )

        self._add_image('{}/AE/A'.format(mode), self.A_batch[0])
        self._add_image('{}/AE/B'.format(mode), self.B_batch[0])
        self._add_image('{}/AE/A_to_B'.format(mode), self.A_to_B[0])
        self._add_image('{}/AE/B_to_A'.format(mode), self.B_to_A[0])

        self._add_image('{}/AE/AB_attn'.format(mode), self.AB_attn[0])
        self._add_image('{}/AE/BA_attn'.format(mode), self.BA_attn[0])
        return

    def GAN_logging(self, mode):
        # Recording only the last batch loss in a epoch
        self.writer.add_scalar(
            '{}/GAN/loss_G'.format(mode), self.loss_G.item(), self.global_step
        )
        self.writer.add_scalar(
            '{}/GAN/loss_ad'.format(mode), self.loss_GAN_A_to_B.item() + self.loss_GAN_B_to_A.item(),
            self.global_step
        )
        self.writer.add_scalar(
            '{}/GAN/loss_cycle'.format(mode), self.loss_cycle_ABA.item() + self.loss_cycle_BAB.item(),
            self.global_step
        )
        self.writer.add_scalar(
            '{}/GAN/loss_id'.format(mode), self.loss_identity_A.item() + self.loss_identity_B.item(),
            self.global_step
        )
        self.writer.add_scalar(
            '{}/GAN/loss_D'.format(mode), self.loss_D.item(), self.global_step
        )
        self.writer.add_scalar(
            '{}/GAN/W'.format(mode), self.W_A.item() + self.W_B.item(), self.global_step
        )
        if mode == 'train':
            self.writer.add_scalar(
                '{}/GAN/GP'.format(mode), self.GP_A.item() + self.GP_B.item(), self.global_step
            )

        self._add_image('{}/GAN/A'.format(mode), self.A_batch[0])
        self._add_image('{}/GAN/B'.format(mode), self.B_batch[0])
        self._add_image('{}/GAN/A_to_B'.format(mode), self.A_to_B[0])
        self._add_image('{}/GAN/B_to_A'.format(mode), self.B_to_A[0])
        self._add_image('{}/GAN/A_to_B_to_A'.format(mode), self.A_to_B_to_A[0])
        self._add_image('{}/GAN/B_to_A_to_B'.format(mode), self.B_to_A_to_B[0])
        self._add_image('{}/GAN/A_to_A'.format(mode), self.A_to_A[0])
        self._add_image('{}/GAN/B_to_B'.format(mode), self.B_to_B[0])

        self._add_image('{}/GAN/AB_attn'.format(mode), self.AB_attn[0])
        self._add_image('{}/GAN/BA_attn'.format(mode), self.BA_attn[0])

        return

    def train(self):
        self.gen_A_to_B.train()
        self.gen_B_to_A.train()
        self.dis_A.train()
        self.dis_B.train()

        self.epoch_loss = 0
        for idx, (A_id, A_batch, B_id, B_batch) in enumerate(self.train_loader):
            self.A_batch, self.B_batch = A_batch.to(self.device), B_batch.to(self.device)

            if self.pre_train:
                # AE pre-train
                self.AE_step(mode='train')
                self.epoch_loss += self.loss_AE.item()
                print('gs = %d, loss = %f' % (self.global_step, self.loss_AE.item()))
            else:
                # Train Generator
                self.G_step(mode='train')
                for _ in range(2):
                    # Train Discriminator
                    self.D_step(mode='train')

                print('gs = %d, loss_G = %f, loss_ad = %f, loss_cycle = %f, loss_id = %f' %
                    (self.global_step,
                     self.loss_G.item(),
                     self.loss_GAN_A_to_B.item() + self.loss_GAN_B_to_A.item(),
                     self.loss_cycle_ABA.item() + self.loss_cycle_BAB.item(),
                     self.loss_identity_A.item() + self.loss_identity_B.item()
                    )
                )

                print('gs = %d, loss_D = %f, W = %f, GP = %f' %
                    (self.global_step,
                     self.loss_D.item(),
                     self.W_A.item() + self.W_B.item(),
                     self.GP_A.item() + self.GP_B.item(),
                    )
                )

            # Saving or not
            self.global_step += 1
            if self.global_step % self.ckpt_interval == 0:
                self.save_ckpt()

        if self.pre_train:
            self.AE_logging(mode='train', batch_num=idx+1)
        else:
            self.GAN_logging(mode='train')
        self.epoch += 1

        return

    def eval(self):
        self.gen_A_to_B.eval()
        self.gen_B_to_A.eval()
        self.dis_A.eval()
        self.dis_B.eval()

        self.eval_loss = 0
        with torch.no_grad():
            for idx, (A_id, A_batch, B_id, B_batch) in enumerate(self.eval_loader):
                self.A_batch, self.B_batch = A_batch.to(self.device), B_batch.to(self.device)

                if self.pre_train:
                    self.AE_step(mode='eval')
                    self.eval_loss += self.loss_AE.item()
                    # only use 100 samples for eval
                    if idx == 100:
                        break
                else:
                    self.G_step(mode='eval')
                    self.D_step(mode='eval')
                    # only use 1 sample for eval
                    break

            if self.pre_train:
                self.AE_logging(mode='eval', batch_num=idx+1)
            else:
                self.GAN_logging(mode='eval')

        return
