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

from .models import STAR_Generator, STAR_Discriminator, OnehotEncoder
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

        self.save_prefix = "starGAN_log"

        self.mode = mode
        if mode == 'train':
            self.train_loader, self.class_num = self.get_dataset(self.train_meta_path)
            self.gen = self.build_gen()
            self.dis = self.build_dis()
            self.gen_optimizer, self.dis_optimizer = self.build_optimizer()
            self.eval_loader = self.get_dataset(self.eval_meta_path)
            self.log_dir = os.path.join(config['path']['uppt']['log_dir'], self.save_prefix)
            self.writer = SummaryWriter(self.log_dir)
            self.pre_train = args.pre_train
        elif mode == 'test':
            self.test_loader, self.class_num = self.get_dataset(self.test_meta_path)
            self.gen = self.build_gen()
        self.one_hot_encoder = self.build_one_hot()

        self.save_dir = os.path.join(config['path']['uppt']['save_dir'], self.save_prefix)
        self.uppt_output_dir = os.path.join(config['path']['uppt']['output_dir'], self.save_prefix)

        # attempt to load or set gs and epoch to 0
        self.load_ckpt()

    def get_dataset(self, meta_path):
        dataset = STAR_VCTKDataset(
            feat_dir=self.feat_dir,
            meta_path=meta_path,
            dict_path=self.dict_path,
            phn_hat_dir=self.phn_hat_dir,
            max_len=self.max_len,
            mode=self.mode
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=True if self.mode == 'train' else False,
            num_workers=self.num_workers,
            collate_fn=dataset._collate_fn, pin_memory=True
        )
        return dataloader, dataset.class_num

    def build_gen(self):
        gen = STAR_Generator(
                input_dim=self.phn_dim, r=3, dropout_rate=0.5,
                prenet_hidden_dims=[256,128], K=16,
                conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=128,
                max_decode_len=self.max_len,
                class_num=self.class_num
            ).to(self.device)
        return gen

    def build_dis(self):
        dis = STAR_Discriminator(
                input_dim=self.phn_dim,
                input_len=self.max_len,
                class_num=self.class_num
              ).to(self.device)
        return dis

    def build_one_hot(self):
        one_hot_encoder = OnehotEncoder(self.class_num)
        return one_hot_encoder

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
        return gen_optimizer, dis_optimizer

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
        self.A_to_B, self.AB_attn = self.gen(
            self.A_batch, self.tgt_group_one_hot, teacher_force_rate=tf_rate)
        self.B_to_A, self.BA_attn = self.gen(
            self.B_batch, self.src_group_one_hot, teacher_force_rate=tf_rate)
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
        self.A_to_B, self.AB_attn = self.gen(
            self.A_batch, self.tgt_group_one_hot, teacher_force_rate=tf_rate)
        self.v_B_fake, self.clf_B_fake = self.dis(self.A_to_B)
        self.loss_GAN_A_to_B = -1*(self.v_B_fake).mean()

        self.B_to_A, self.BA_attn = self.gen(
            self.B_batch, self.src_group_one_hot, teacher_force_rate=tf_rate)
        self.v_A_fake, self.clf_A_fake = self.dis(self.B_to_A)
        self.loss_GAN_B_to_A = -1*(self.v_A_fake).mean()

        self.loss_clf_A_forG = self._xent_loss(self.clf_A_fake, self.src_group_one_hot)
        self.loss_clf_B_forG = self._xent_loss(self.clf_B_fake, self.tgt_group_one_hot)

        # Cycle-consistency Loss
        self.A_to_B_to_A, _ = self.gen(
            self.A_to_B, self.src_group_one_hot, teacher_force_rate=tf_rate)
        self.B_to_A_to_B, _ = self.gen(
            self.B_to_A, self.tgt_group_one_hot, teacher_force_rate=tf_rate)
        self.loss_cycle_ABA = self._xent_loss(self.A_to_B_to_A, self.A_batch)
        self.loss_cycle_BAB = self._xent_loss(self.B_to_A_to_B, self.B_batch)

        # Identity Loss
        self.A_to_A, _ = self.gen(
            self.A_batch, self.src_group_one_hot, teacher_force_rate=tf_rate)
        self.B_to_B, _ = self.gen(
            self.B_batch, self.tgt_group_one_hot, teacher_force_rate=tf_rate)
        self.loss_identity_A = self._xent_loss(self.A_to_A, self.A_batch)
        self.loss_identity_B = self._xent_loss(self.B_to_B, self.B_batch)

        # Total Loss
        self.loss_G = self.loss_GAN_A_to_B + self.loss_GAN_B_to_A + \
                 self.loss_cycle_ABA*10 + self.loss_cycle_BAB*10 + \
                 self.loss_identity_A*5 + self.loss_identity_B*5 + \
                 self.loss_clf_A_forG*10 + self.loss_clf_B_forG*10

        if mode == 'train':
            self.loss_G.backward()
            self.gen_optimizer.step()

        return

    def D_step(self, mode):
        ##### Discriminators #####
        if mode == 'train':
            self.dis_optimizer.zero_grad()

        self.v_A_real, self.clf_A_real = self.dis(self.A_batch)
        self.v_B_real, self.clf_B_real = self.dis(self.B_batch)
        self.v_A_fake, _ = self.dis(self.B_to_A.detach())
        self.v_B_fake, _ = self.dis(self.A_to_B.detach())
        self.W_A = (self.v_A_real - self.v_A_fake).mean()
        self.W_B = (self.v_B_real - self.v_B_fake).mean()

        self.loss_clf_A_forD = self._xent_loss(self.clf_A_real, self.src_group_one_hot)
        self.loss_clf_B_forD = self._xent_loss(self.clf_B_real, self.tgt_group_one_hot)

        if mode == 'train':
            # Gradient Penalty
            cur_batch_size = self.A_batch.shape[0]
            alpha = torch.rand(cur_batch_size, 1, 1).to(self.device)
            self.A_inter = alpha * self.B_to_A.detach() + (1.0-alpha) * self.A_batch
            self.A_inter = Variable(self.A_inter, requires_grad=True).to(self.device)
            self.v_A_inter, _ = self.dis(self.A_inter)
            gradient_A = torch.autograd.grad(
                self.v_A_inter, self.A_inter,
                grad_outputs=torch.ones(self.v_A_inter.size()).to(self.device),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            self.GP_A = ((gradient_A.norm(2) -1) ** 2).mean()

            alpha = torch.rand(cur_batch_size, 1, 1).to(self.device)
            self.B_inter = alpha * self.A_to_B.detach() + (1.0-alpha) * self.B_batch
            self.B_inter = Variable(self.B_inter, requires_grad=True).to(self.device)
            self.v_B_inter, _ = self.dis(self.B_inter)
            gradient_B = torch.autograd.grad(
                self.v_B_inter, self.B_inter,
                grad_outputs=torch.ones(self.v_B_inter.size()).to(self.device),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            self.GP_B = ((gradient_B.norm(2) -1) ** 2).mean()

            # Total Loss
            self.loss_D = -1*(self.W_A + self.W_B) + \
                10*(self.GP_A + self.GP_B) + 10*(self.loss_clf_A_forD + self.loss_clf_B_forD)
            self.loss_D.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.dis.parameters(), 5.)
            self.dis_optimizer.step()

        else:
            # for recording W distance only
            self.loss_D = -1*(self.W_A + self.W_B) + 10*(self.loss_clf_A_forD + self.loss_clf_B_forD)

        return

    def _add_image(self, name, data):
        self.writer.add_image(name,
            torch.t(data).detach().cpu().numpy(),
            self.global_step, dataformats='HW'
        )
        return

    def AE_logging(self, mode):
        if mode == 'train':
            self.writer.add_scalar(
                'train/AE/training_loss',
                self.loss_AE,
                self.global_step
            )
        else:
            self.writer.add_scalar(
                'eval/AE/epoch_loss',
                self.eval_loss,
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
        self._add_image('{}/GAN/A_to_B'.format(mode), self.A_to_B[0])
        self._add_image('{}/GAN/B'.format(mode), self.B_batch[0])
        self._add_image('{}/GAN/B_to_A'.format(mode), self.B_to_A[0])

        self._add_image('{}/GAN/A_to_B_to_A'.format(mode), self.A_to_B_to_A[0])
        self._add_image('{}/GAN/B_to_A_to_B'.format(mode), self.B_to_A_to_B[0])
        self._add_image('{}/GAN/A_to_A'.format(mode), self.A_to_A[0])
        self._add_image('{}/GAN/B_to_B'.format(mode), self.B_to_B[0])

        self._add_image('{}/GAN/AB_attn'.format(mode), self.AB_attn[0])
        self._add_image('{}/GAN/BA_attn'.format(mode), self.BA_attn[0])

        return

    def train(self):
        self.gen.train()
        self.dis.train()

        for idx, (src_group, A_id, A_batch, tgt_group, B_id, B_batch) in enumerate(self.train_loader):
            self.A_batch, self.B_batch = A_batch.to(self.device), B_batch.to(self.device)
            self.src_group_one_hot, self.tgt_group_one_hot = \
                self.one_hot_encoder(src_group).to(self.device), \
                self.one_hot_encoder(tgt_group).to(self.device)

            if self.pre_train:
                # AE pre-train
                self.AE_step(mode='train')
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

            self.global_step += 1
            if self.global_step % self.summ_interval == 0:
                if self.pre_train:
                    self.AE_logging(mode='train')
                else:
                    self.GAN_logging(mode='train')
            # Saving or not
            if self.global_step % self.ckpt_interval == 0:
                self.save_ckpt()

        self.epoch += 1

        return

    def eval(self):
        self.gen.eval()
        self.dis.eval()

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
                self.eval_loss /= (idx+1)
                self.AE_logging(mode='eval')
            else:
                self.GAN_logging(mode='eval')

        return
