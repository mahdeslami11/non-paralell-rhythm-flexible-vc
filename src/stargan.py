import os
import itertools
import pickle
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from .models import Generator, Discriminator, OnehotEncoder
from .dataset import STAR_VCTKDataset
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

class STAR_Solver(Solver):
    def __init__(self, config, args, mode='train'):
        super(STAR_Solver, self).__init__(config, args)

        self.phn_hat_dir = config['path']['ppr']['output_dir']
        self.phn_dim = config['text']['phn_dim']

        self.lr = config['model']['uppt']['lr']
        self.optimizer_type = config['model']['uppt']['type']
        self.betas = [float(x) for x in config['model']['uppt']['betas'].split(',')]
        self.weight_decay = config['model']['uppt']['weight_decay']
        self.max_len = config['model']['uppt']['max_len']

        self.ids = ['p231', 'p256', 'p265', 'p340']

        self.mode = mode
        self.gen = self.build_gen()
        if mode == 'train':
            self.dis = self.build_dis()
            self.clf = self.build_clf()
            self.gen_optimizer, self.dis_optimizer, self.clf_optimizer = self.build_optimizer()
            self.train_loader = self.get_dataset(self.train_meta_path)
            self.eval_loader = self.get_dataset(self.eval_meta_path)
            self.log_dir = os.path.join(config['path']['uppt']['log_dir'], 'star')
            self.writer = SummaryWriter(self.log_dir)
        elif mode == 'test':
            self.test_loader = self.get_dataset(self.test_meta_path)

        self.save_dir = os.path.join(config['path']['uppt']['save_dir'], 'star')
        self.uppt_output_dir = os.path.join(config['path']['uppt']['output_dir'], 'star')

        # attempt to load or set gs and epoch to 0
        self.load_ckpt()

    def get_dataset(self, meta_path):
        dataset = STAR_VCTKDataset(
            feat_dir=self.feat_dir,
            meta_path=meta_path,
            dict_path=self.dict_path,
            phn_hat_dir=self.phn_hat_dir,
            ids=self.ids,
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
        gen = Generator(
            input_dim=self.phn_dim, r=5, dropout_rate=0.5,
            prenet_hidden_dims=[256,128], K=16,
            conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=128,
            max_decode_len=self.max_len
        ).to(self.device)

        return gen

    def build_dis(self):
        dis = Discriminator(input_dim=self.phn_dim, input_len=self.max_len).to(self.device)
        return dis

    def build_optimizer(self):
        optimizer = getattr(torch.optim, self.optimizer_type)
        gen_optimizer = optimizer(
            itertools.chain(self.gen.parameters()),
            lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        dis_optimizer = optimizer(
            itertools.chain(self.dis.parameters()),
            lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        clf_optimizer = optimizer(
            itertools.chain(self.clf.parameters()),
            lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )

        return gen_optimizer, dis_optimizer, clf_optimizer

    def save_ckpt(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        checkpoint_path = os.path.join(
            self.save_dir, "model.ckpt-{}.pt".format(self.global_step)
        )
        torch.save({
            "gen": self.gen.state_dict(),
            "dis": self.dis.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict(),
            "dis_optimizer": self.dis_optimizer.state_dict(),
            "clf_optimizer": self.clf_optimizer.state_dict(),
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
            self.gen.load_state_dict(checkpoint['gen'])
            if self.mode == 'train':
                self.dis.load_state_dict(checkpoint['dis'])
                self.clf.load_state_dict(checkpoint['clf'])
                self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
                self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
                self.clf_optimizer.load_state_dict(checkpoint['clf_optimizer'])
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
        tf_rate = 1. if mode == 'train' else 0.

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
        self.W_A = (self.v_A_real - self.v_A_fake).mean()
        self.W_B = (self.v_B_real - self.v_B_fake).mean()

        # Gradient Penalty
        alpha = torch.rand(self.batch_size, 1, 1).to(self.device)
        self.A_inter = alpha * self.B_to_A + (1.0-alpha) * self.A_batch
        self.v_A_inter = self.dis_A(self.A_inter)
        gradient_A = torch.autograd.grad(
            self.v_A_inter, self.A_inter,
            grad_outputs=torch.ones(self.v_A_inter.size()).to(self.device),
            create_graph=True, retain_graph=True, only_inputs=True
        )
        self.GP_A = ((gradient_A.norm(2) -1) ** 2).mean()

        alpha = torch.rand(self.batch_size, 1, 1).to(self.device)
        self.B_inter = alpha * self.A_to_B + (1.0-alpha) * self.B_batch
        self.v_B_inter = self.dis_B(self.B_inter)
        gradient_B = torch.autograd.grad(
            self.v_B_inter, self.B_inter,
            grad_outputs=torch.ones(self.v_B_inter.size()).to(self.device),
            create_graph=True, retain_graph=True, only_inputs=True
        )
        self.GP_B = ((gradient_B.norm(2) -1) ** 2).mean()

        # Total Loss
        self.loss_D = -1*(self.W_A + self.W_B) + 10*(self.GP_A + self.GP_B)
        if mode == 'train':
            self.loss_D.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(
                itertools.chain(self.dis_A.parameters(), self.dis_B.parameters()), 5.
            )
            self.dis_optimizer.step()

        return

    def _add_image(self, name, data):
        self.writer.add_image(name,
            torch.t(data).detach().cpu().numpy(),
            self.epoch, dataformats='HW'
        )
        return

    def AE_logging(self, mode, batch_num):
        if mode == 'train':
            print('[epoch %d] training_loss: %.6f' % (self.epoch, self.epoch_loss/batch_num))
            self.writer.add_scalar(
                'train/AE/epoch_training_loss',
                self.epoch_loss/batch_num,
                self.epoch
            )
        else:
            print('[epoch %d] eval_loss: %.6f' % (self.epoch, self.eval_loss/batch_num))
            self.writer.add_scalar(
                'eval/AE/epoch_training_loss',
                self.eval_loss/batch_num,
                self.epoch
            )

        self._add_image('{}/AE/A'.format(mode), self.A_batch[0])
        self._add_image('{}/AE/B'.format(mode), self.B_batch[0])
        self._add_image('{}/AE/A_to_B'.format(mode), self.A_to_B[0])
        self._add_image('{}/AE/B_to_A'.format(mode), self.B_to_A[0])

        self._add_image('{}/AE/AB_attn'.format(mode), self.AB_attn[0])
        self._add_image('{}/AE/BA_attn'.format(mode), self.BA_attn[0])
        return

    def GAN_logging(self):
        #TODO
        return

    def train(self):
        self.gen_A_to_B.train()
        self.gen_B_to_A.train()
        self.dis_A.train()
        self.dis_B.train()

        self.epoch_loss = 0
        for idx, (A_id, A_batch, B_id, B_batch) in enumerate(self.train_loader):
            self.A_batch, self.B_batch = A_batch.to(self.device), B_batch.to(self.device)

            if self.global_step < 20000:
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

            # Saving or not
            self.global_step += 1
            if self.global_step % self.ckpt_interval == 0:
                self.save_ckpt()

        if self.global_step < 2000:
            self.AE_logging(mode='train', batch_num=idx+1)
        else:
            self.GAN_logging(mode='train', batch_num=idx+1)
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

                if self.global_step < 20000:
                    self.AE_step(mode='eval')
                    self.eval_loss += self.loss_AE.item()
                else:
                    #TODO
                    self.G_step(mode='eval')
                    self.D_step(mode='eval')

                # only use 100 samples for eval
                if idx == 100:
                    break

            if self.global_step < 20000:
                self.AE_logging(mode='eval', batch_num=idx+1)
            else:
                self.GAN_logging(mode='eval', batch_num=idx+1)
        
        return
