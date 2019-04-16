import os
import pickle
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from models import PPR, PPTS, Generator, Discriminator, OnehotEncoder
from dataset import PPR_VCTKDataset, PPTS_VCTKDataset, UPPT_VCTKDataset

class Solver(object):
    def __init__(self, config, args):
        self.use_gpu = config['solver']['use_gpu'] and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_gpu else torch.device('cpu')
        self.config = config
        self.args = args

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
        self.train_loader = self.get_dataset(self.train_meta_path)
        self.eval_loader = self.get_dataset(self.eval_meta_path)
        self.test_loader = self.get_dataset(self.test_meta_path)
        self.model = self.build_model()
        if mode == 'train':
            self.optimizer = self.build_optimizer()
            self.one_hot_encoder = OnehotEncoder(self.phn_dim)

        self.log_dir = config['path']['log_dir']
        self.save_dir = config['path']['save_dir']
        self.writer = SummaryWriter(self.log_dir)
        self.ppr_output_dir = config['path']['ppr_output_dir']
        
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
            shuffle=True if self.mode == 'train' else False,
            num_workers=self.num_workers,
            collate_fn=dataset._collate_fn, pin_memory=True
        )
        return dataloader

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
        checkpoint_path = os.path.join(
            self.save_dir, "model.ckpt-{}.pt".format(self.global_step)
        )
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch
        }, checkpoint_path)
        with open(os.path.join(self.save_dir, "checkpoint"), "w") as f:
            f.write("model.ckpt-{}".format(self.global_step))
        return

    def load_ckpt(self):
        checkpoint_list = os.path.join(self.save_dir, 'checkpoint')
        if os.path.exists(checkpoint_list):
            checkpoint_filename = open(checkpoint_list).readline().strip()
            checkpoint_path = os.path.join(hp.save_dir, "{}.pt".format(checkpoint_filename))
            if self.use_gpu:
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model'])
            if self.mode == 'train':
                self.optimizer.load_state_dict(checkpoint['model'])
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']
        else:
            self.global_step = 0
            self.epoch = 0
        return

    def _label_smoothing(self, label_hat, label_batch):
        # label_hat: log-likelihood, label_batch: list of int
        eps = 0.05
        print(label_batch)
        exit()
        one_hot = self.one_hot_encoder(label_batch)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (label_hat.shape[-1] - 1)
        one_hot = one_hot.to(self.device)
        loss = -(one_hot * label_hat).sum(-1).mean()
        return loss

    def _calc_acc(label_hat, label_batch):
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
        for idx, (mel_batch, label_batch) in enumerate(self.train_loader):
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
                    '[GS=%3d, %d, %3d] loss: %.6f' % \
                    (self.global_step+1, self.epoch+1, idx+1, loss.item())
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
        print('[epoch %d] training_loss: %.6f' % (self.epoch + 1, epoch_loss))
        self.writer.add_scalar('train/epoch_training_loss', epoch_loss, epoch)
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
            for idx, (mel_batch, label_batch) in enumerate(eval_loader):
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
        print('[eval %d] eval_loss: %.6f' % (epoch+1, eval_loss))
        print('[eval %d] eval_acc: %.6f' % (epoch+1, avg_acc))

        self.writer.add_scalar('eval/eval_loss', eval_loss, self.epoch)
        self.writer.add_scalar('eval/accuracy', avg_acc, self.epoch)
        self.writer.add_image('eval/label_hat',
            torch.t(torch.exp(label_hat[0])).detach().cpu().numpy(),
            self.epoch, dataformats='HW'
        )

        return

    def save_label_hat(self, f_id_list, label_hat):
        for idx in len(f_id_list):
            out_name = "{}_phn_hat.pkl".format(f_id_list[idx])
            out_path = os.path.join(self.ppr_output_dir, out_name)
            with open(out_path, 'wb') as f:
                pickle.dump(label_hat[idx], f)
        return

    def test(self):
        if not os.path.exists(self.ppr_output_dir):
            os.makedirs(self.ppr_output_dir)
        self.model.eval()
        with torch.no_grad():
            for (f_id_list, mel_batch, label_batch) in test_loader:
                mel_batch, label_batch = mel_batch.to(self.device), label_batch.to(self.device)
                label_hat = self.model(mel_batch).detach().cpu().numpy()
                self.save_label_hat(f_id_list, label_hat)
        return

