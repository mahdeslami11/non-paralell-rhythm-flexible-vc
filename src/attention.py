import torch
import torch.nn as nn

class AttnLayer(nn.Module):
    """
    Input : decoder_state: [batch, 1, dec_hidden_dim]
            encoder_feature: [batch, enc_len, enc_hidden_dim]
    Output: attn_weight: [batch, enc_len]
            context: [batch, enc_hidden_dim]
    TODO: mask encoder padding part
    """
    def __init__(self, attn_mlp_dim, enc_feat_dim):

        super(AttnLayer, self).__init__()
        self.attn_mlp_dim = attn_mlp_dim
        self.enc_feat_dim = enc_feat_dim
        self.softmax = nn.Softmax(dim=-1)

        self.mlp_preprocess_input = True

        if self.mlp_preprocess_input:
            self.preprocess_mlp_dim = 256
            self.phi = nn.Linear(self.enc_feat_dim, self.preprocess_mlp_dim)
            self.psi = nn.Linear(self.enc_feat_dim, self.preprocess_mlp_dim)
            self.activate = nn.ReLU()

    def forward(self, decoder_state, encoder_feature):
        if self.mlp_preprocess_input:
            if self.activate:
                comp_decoder_state = self.activate(self.phi(decoder_state))
                comp_encoder_feature = self.activate(self.psi(encoder_feature))
            else:
                comp_decoder_state = self.phi(decoder_state)
                comp_encoder_feature = self.psi(encoder_feature)
        else:
            comp_decoder_state = decoder_state
            comp_encoder_feature = encoder_feature
        energy = torch.bmm(comp_decoder_state, comp_encoder_feature.transpose(1, 2)).squeeze(dim=1)
        attn_weight = self.softmax(energy)
        context = torch.bmm(attn_weight.unsqueeze(dim=1), encoder_feature).squeeze(dim=1)

        return attn_weight, context

class LocAwareAttnLayer(nn.Module):
    '''
    implementation of: https://arxiv.org/pdf/1506.07503.pdf
    Input : decoder_state: [batch, 1, dec_hidden_dim]
            encoder_feature: [batch, enc_len, enc_hidden_dim]
            last_align: [batch, enc_len]
    Output: attn_weight: [batch, enc_len]
            context: [batch, enc_hidden_dim]
    TODO: mask encoder padding part
    '''
    def __init__(self, dec_hidden_dim, enc_feat_dim, conv_dim, attn_dim, smoothing=False):
        super(LocAwareAttnLayer, self).__init__()
        self.attn_dim = attn_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.conv_dim = conv_dim
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.conv_dim, kernel_size=3, padding=1)

        self.W = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.V = nn.Linear(enc_feat_dim, attn_dim, bias=False)
        self.U = nn.Linear(conv_dim, attn_dim, bias=False)
        self.b = nn.Parameter(torch.rand(attn_dim))
        self.w = nn.Linear(attn_dim, 1, bias=False)

        self.tanh = nn.Tanh()
        self.smoothing = smoothing
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_state, encoder_feature, last_align):
        # conv_feat: [batch, enc_len, conv_dim]
        conv_feat = torch.transpose(self.conv(last_align.unsqueeze(dim=1)), 1, 2)
        # energy: [batch, enc_len]
        energy = self.w(self.tanh(
            self.W(decoder_state)
            + self.V(encoder_feature)
            + self.U(conv_feat)
            + self.b
        )).squeeze(dim=-1)

        if self.smoothing:
            energy = torch.sigmoid(energy)
            attn_weight = torch.div(energy, energy.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn_weight = self.softmax(energy)

        context = torch.bmm(attn_weight.unsqueeze(dim=1), encoder_featuret).squeeze(dim=1)

        return attn_weight, context
