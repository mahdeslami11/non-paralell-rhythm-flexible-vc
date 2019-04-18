import numpy as np
import torch
import torch.nn as nn

from .modules import Prenet, CBHG, GLU, CIG_block
from .attention import AttnLayer, LocAwareAttnLayer

class OnehotEncoder(nn.Module):
    def __init__(self, n_class):
        super(OnehotEncoder, self).__init__()
        self.encoder = nn.Embedding(n_class, n_class)
        self.encoder.weight.data = torch.eye(n_class)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, batch):
        return self.encoder(batch.cpu())

class PPR(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5,
                 prenet_hidden_dims=[256, 128], K=16,
                 conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=128):
        super(PPR, self).__init__()
        self.prenet = Prenet(input_dim, dropout_rate, prenet_hidden_dims)
        self.cbhg = CBHG(
            prenet_hidden_dims[-1], K,
            conv1d_bank_hidden_dim, conv1d_projections_hidden_dim, gru_dim
        )
        self.output_trans = nn.Linear(gru_dim*2, output_dim)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_x):
        h1 = self.prenet(input_x)
        h2 = self.cbhg(h1)
        logits = self.output_trans(h2)
        output = self.softmax(logits)
        return output

class PPTS(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5,
                 prenet_hidden_dims=[256, 128], K=16,
                 conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=256):
        super(PPTS, self).__init__()
        self.prenet = Prenet(input_dim, dropout_rate, prenet_hidden_dims)
        self.cbhg = CBHG(
            prenet_hidden_dims[-1], K,
            conv1d_bank_hidden_dim, conv1d_projections_hidden_dim, gru_dim
        )
        self.gru = nn.GRU(2*gru_dim, gru_dim, 1, batch_first=True, bidirectional=True)
        self.output_trans = nn.Linear(gru_dim*2, output_dim)

    def forward(self, input_x):
        h1 = self.prenet(input_x)
        h2 = self.cbhg(h1)
        h2, _ = self.gru(h2)
        output = self.output_trans(h2)
        return output

class UPPT_Encoder(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5, prenet_hidden_dims=[256, 128], K=16,
                 conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=128):
        super(UPPT_Encoder, self).__init__()
        self.prenet = Prenet(input_dim, dropout_rate, prenet_hidden_dims)
        self.cbhg = CBHG(
            prenet_hidden_dims[-1], K,
            conv1d_bank_hidden_dim, conv1d_projections_hidden_dim, gru_dim
        )

    def forward(self, input_x):
        h1 = self.prenet(input_x)
        output = self.cbhg(h1)
        return output

class UPPT_Decoder(nn.Module):
    def __init__(self, input_dim, enc_feat_dim=256,
                 attn_dim=128, num_layer=1, gru_dim=256,
                 loc_aware=False, max_decode_len=500):
        super(UPPT_Decoder, self).__init__()
        self.input_dim = input_dim
        self.gru = nn.GRU(input_dim+enc_feat_dim, gru_dim, 1, batch_first=True)
        self.loc_aware = loc_aware
        self.max_decode_len = max_decode_len

        if self.loc_aware:
            self.attention = LocAwareAttnLayer(
                dec_hidden_dim=gru_dim,
                enc_feat_dim=enc_feat_dim,
                conv_dim=64,
                attn_dim=attn_dim,
                smoothing=False
            )
        else:
            self.attention = AttnLayer(
                attn_mlp_dim=attn_dim,
                enc_feat_dim=enc_feat_dim
            )

        self.output_trans = nn.Linear(gru_dim+enc_feat_dim, input_dim)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward_step(self, input_x, last_hidden_state, enc_feat, last_context, last_attn_weight):
        # concat last context with input
        input_mix = torch.cat([input_x, last_context.unsqueeze(dim=1)], dim=-1)
        # feed to RNN
        rnn_output, hidden_state = self.gru(input_mix, last_hidden_state)
        # compute attn context for next step
        if self.loc_aware:
            attn_weight, context = self.attention(rnn_output, enc_feat, last_attn_weight)
        else:
            attn_weight, context = self.attention(rnn_output, enc_feat)
        # and also for this step's output
        concat_feature = torch.cat([rnn_output.squeeze(dim=1), context], dim=-1)
        raw_pred = self.output_trans(concat_feature)
        pred = self.softmax(raw_pred)

        return pred, hidden_state, context, attn_weight

    def forward(self, enc_feat, ground_truth=None, teacher_force_rate=0.5):
        if ground_truth is None:
            # eval or infer time
            teacher_force = False
            max_step = self.max_decode_len
        else:
            # sampling
            teacher_force = True if np.random.random_sample() < teacher_force_rate else False
            if teacher_force is True:
                max_step = ground_truth.size()[1]
            else:
                max_step = self.max_decode_len

        batch_size = enc_feat.shape[0]
        hidden_state = None
        context = enc_feat.new_zeros((batch_size, enc_feat.shape[-1]))
        attn_weight = enc_feat.new_zeros((batch_size, enc_feat.shape[1]))
        pred_seq = []
        attn_record = []

        for step in range(max_step):
            if step == 0:
                if teacher_force:
                    input_x = ground_truth[:,0,:]
                else:
                    # manually setting GO symbol for free run
                    input_x = enc_feat.new_zeros((batch_size, self.input_dim))
            else:
                if teacher_force:
                    input_x = ground_truth[:,step:step+1,:].squeeze(1)
                else:
                    input_x = pred

            # input_x: [batch, input_dim]
            input_x = input_x.unsqueeze(dim=1)
            pred, hidden_state, context, attn_weight = \
                self.forward_step(
                    input_x, hidden_state, enc_feat,
                    context, attn_weight
                )
            pred_seq.append(pred)
            attn_record.append(attn_weight)

        pred_seq = torch.stack(pred_seq, dim=1)
        attn_record = torch.stack(attn_record, dim=1)

        return pred_seq, attn_record

class Generator(nn.Module):
    def __init__(self, input_dim=70, dropout_rate=0.5,
                 prenet_hidden_dims=[256,128], K=16,
                 conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=128):
        super(Generator, self).__init__()
        self.encoder = UPPT_Encoder(
                input_dim=input_dim, dropout_rate=0.5,
                prenet_hidden_dims=[256,128], K=16,
                conv1d_bank_hidden_dim=128, conv1d_projections_hidden_dim=128, gru_dim=128
            )
        self.decoder = UPPT_Decoder(
                input_dim=input_dim, enc_feat_dim=2*gru_dim,
                attn_dim=128, num_layer=1, gru_dim=256, loc_aware=False, max_decode_len=500
            )

    def forward(self, input_x, teacher_force_rate=0.5):
        enc_feat = self.encoder(input_x)
        pred_seq, attn_record = self.decoder(
            enc_feat, ground_truth=input_x, teacher_force_rate=0.5
        )
        return pred_seq, attn_record

class Discriminator(nn.Module):
    def __init__(self, input_dim, input_len, is_WGAN=True):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.conv1d = nn.Conv1d(
            input_dim, 128,
            kernel_size=3, stride=2, padding=3//2
        )
        self.glu = GLU(128)
        self.CIG_blocks_dims = [(128,256),(256,512),(512,1024)]
        self.CIG_kernels = [3, 3, 6]
        self.CIG_strides = [2, 2, 2]
        self.CIG_blocks = nn.ModuleList(
            [CIG_block(in_dim, out_dim, kernel_size=k, stride=s)
                for (in_dim, out_dim), k, s in \
                    zip(self.CIG_blocks_dims, self.CIG_kernels, self.CIG_strides)]
        )

        self.reduce_len = int(((input_len-3)/2)+1) # first conv1d
        for k, s in zip(self.CIG_kernels, self.CIG_strides):
            self.reduce_len = int(((self.reduce_len-k)/s)+1) # 3*CIG_block
        self.output_trans = nn.Linear(self.CIG_blocks_dims[-1][1]*self.reduce_len, 1)

        self.is_WGAN = is_WGAN
        if not self.is_WGAN:
            self.activation = nn.Sigmoid()

    def forward(self, input_x):
        # input_x: [batch, len, input_dim] -> [batch, input_dim, len]
        input_x_t = input_x.transpose(1, 2) if input_x.shape[-1] == self.input_dim else input_x
        h1 = self.glu(self.conv1d(input_x_t))
        print(h1.shape)
        h2 = self.CIG_blocks[0](h1)
        print(h2.shape)
        h3 = self.CIG_blocks[1](h2)
        print(h3.shape)
        h4 = self.CIG_blocks[2](h3)
        print(h4.shape)
        h4 = torch.flatten(h4, 1, -1)
        output = self.output_trans(h4)
        if not self.is_WGAN:
            output = self.activation(output)
        return output
