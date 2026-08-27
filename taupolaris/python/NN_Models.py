import torch
import torch.nn as nn
import nflows
from nflows.nn import nets
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.lu import LULinear
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, AffineCouplingTransform
from nflows.flows.base import Flow
import time

# Model Definitions

def NormalizingFlow(input_size=8,
                    context_features=6,
                    num_layers=8,
                    num_bins=8,
                    tail_bound=2.0,
                    hidden_size=64,
                    num_blocks=2,
                    batch_norm=False,
                    activation=nn.ReLU()):
    """Creates a normalizing flow model using nflows library."""
    transforms = []

    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_features,
            hidden_features=hidden_size, num_blocks=num_blocks,
            use_batch_norm=batch_norm,
            activation=activation)

    mask = nflows.utils.torchutils.create_mid_split_binary_mask(input_size)

    for i in range(num_layers):
        if i > 0: transforms.append(nflows.transforms.LULinear(input_size))
        transforms.append(ReversePermutation(features=input_size))
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask, create_net, tails='linear', num_bins=num_bins, tail_bound=tail_bound))
        transforms.append(ReversePermutation(features=input_size))
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask, create_net, tails='linear', num_bins=num_bins, tail_bound=tail_bound))

    transform = CompositeTransform(transforms)
    distribution = StandardNormal([input_size])
    flow = Flow(transform, distribution)
    return flow

def MLP(input_size=8,
    num_blocks=3,
    hidden_size=64,
    output_size=8,
    batch_norm=False,
    activation=nn.ReLU()):

    # define a simple feed-forward neural network using resisual blocks
    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features,
            hidden_features=hidden_size, num_blocks=num_blocks,
            use_batch_norm=batch_norm,
            activation=activation)

    net = create_net(input_size, output_size)
    return net

class ParticleTransformerCondition(nn.Module):
    """
    Tokenises reco tau decay products and encodes with a transformer.

    Mode 0 (hadronic, 13 tokens):
      [taup_pi1, taun_pi1, taup_ip, taun_ip, taup_pi2, taun_pi2,
       taup_pi3, taun_pi3, taup_pi0, taun_pi0, MET, taup_sv, taun_sv]
      pi2/pi3/sv masked when is_3prong=0; pi0 masked when haspizero=0.

    Mode 1 (semi-leptonic, 9 tokens):
      [tau1_lep, tau1_lep_ip, tau2_pi1, tau2_ip, tau2_pi2,
       tau2_pi3, tau2_pi0, MET, tau2_sv]
      tau2 pi2/pi3/sv masked when is_3prong=0; tau2_pi0 masked when haspizero=0.

    Mode -1 (any mix of hadronic/leptonic per tau, no filtering/relabeling
    applied upstream, 17 tokens):
      [taup_pi1, taun_pi1, taup_ip, taun_ip, taup_pi2, taun_pi2,
       taup_pi3, taun_pi3, taup_pi0, taun_pi0, MET, taup_sv, taun_sv,
       taup_lep, taun_lep, taup_lep_ip, taun_lep_ip]
      Per-tau, exactly one of {pi1+ip, lep+lep_ip} is present per event,
      selected by that tau's own reco_ishadronic flag (independently for
      taup and taun, since either/both/neither may be leptonic).
      pi2/pi3/sv masked when is_3prong=0; pi0 masked when haspizero=0
      (on top of the hadronic/leptonic mask).
    """

    # mode 0 feature names
    _taup_pi1_feats = ['reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz', 'reco_taup_pi1_e']
    _taun_pi1_feats = ['reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz', 'reco_taun_pi1_e']
    _taup_charged_ip_feats = ['reco_taup_charged_ipx', 'reco_taup_charged_ipy', 'reco_taup_charged_ipz']
    _taun_charged_ip_feats = ['reco_taun_charged_ipx', 'reco_taun_charged_ipy', 'reco_taun_charged_ipz']
    _taup_pi2_feats = ['reco_taup_pi2_px', 'reco_taup_pi2_py', 'reco_taup_pi2_pz', 'reco_taup_pi2_e']
    _taun_pi2_feats = ['reco_taun_pi2_px', 'reco_taun_pi2_py', 'reco_taun_pi2_pz', 'reco_taun_pi2_e']
    _taup_pi3_feats = ['reco_taup_pi3_px', 'reco_taup_pi3_py', 'reco_taup_pi3_pz', 'reco_taup_pi3_e']
    _taun_pi3_feats = ['reco_taun_pi3_px', 'reco_taun_pi3_py', 'reco_taun_pi3_pz', 'reco_taun_pi3_e']
    _taup_pizero_feats = ['reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz', 'reco_taup_pizero1_e']
    _taun_pizero_feats = ['reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz', 'reco_taun_pizero1_e']
    _taup_sv_feats = ['reco_taup_sv_x', 'reco_taup_sv_y', 'reco_taup_sv_z']
    _taun_sv_feats = ['reco_taun_sv_x', 'reco_taun_sv_y', 'reco_taun_sv_z']

    # mode 1 feature names (tau1=lepton, tau2=hadronic)
    _tau1_lep_feats = ['reco_tau1_lep_px', 'reco_tau1_lep_py', 'reco_tau1_lep_pz', 'reco_tau1_lep_e']
    _tau1_lep_ip_feats = ['reco_tau1_lep_ipx', 'reco_tau1_lep_ipy', 'reco_tau1_lep_ipz']
    _tau2_pi1_feats = ['reco_tau2_pi1_px', 'reco_tau2_pi1_py', 'reco_tau2_pi1_pz', 'reco_tau2_pi1_e']
    _tau2_charged_ip_feats = ['reco_tau2_charged_ipx', 'reco_tau2_charged_ipy', 'reco_tau2_charged_ipz']
    _tau2_pi2_feats = ['reco_tau2_pi2_px', 'reco_tau2_pi2_py', 'reco_tau2_pi2_pz', 'reco_tau2_pi2_e']
    _tau2_pi3_feats = ['reco_tau2_pi3_px', 'reco_tau2_pi3_py', 'reco_tau2_pi3_pz', 'reco_tau2_pi3_e']
    _tau2_pizero_feats = ['reco_tau2_pizero1_px', 'reco_tau2_pizero1_py', 'reco_tau2_pizero1_pz', 'reco_tau2_pizero1_e']
    _tau2_sv_feats = ['reco_tau2_sv_x', 'reco_tau2_sv_y', 'reco_tau2_sv_z']

    # mode -1 feature names (taup/taun, either may be leptonic or hadronic)
    _taup_lep_feats = ['reco_taup_lep_px', 'reco_taup_lep_py', 'reco_taup_lep_pz', 'reco_taup_lep_e']
    _taup_lep_ip_feats = ['reco_taup_lep_ipx', 'reco_taup_lep_ipy', 'reco_taup_lep_ipz']
    _taun_lep_feats = ['reco_taun_lep_px', 'reco_taun_lep_py', 'reco_taun_lep_pz', 'reco_taun_lep_e']
    _taun_lep_ip_feats = ['reco_taun_lep_ipx', 'reco_taun_lep_ipy', 'reco_taun_lep_ipz']

    _met_feats = ['reco_met_px', 'reco_met_py']

    # engineered hadronic-current features, one block of 11 per leg -- see
    # engineered_polvec_features.md and acoplanarity_tools.hadronic_current_features
    _HCUR_SUFFIXES = ('re_n', 're_r', 're_k', 'im_n', 'im_r', 'im_k',
                      's1', 's2', 's3', 'm2vis')
    HCUR_BLOCK_SIZE = len(_HCUR_SUFFIXES)

    @classmethod
    def _hcur_feats(cls, tau, block):
        return [f'reco_{tau}_{block}{suf}' for suf in cls._HCUR_SUFFIXES]

    def __init__(self, input_features, leptonic_mode, context_dim=256, d_model=64, nhead=4, num_layers=3, dropout=0.0,
                 legacy_pizero_proj=False, polvec_feature_level=0):
        """
        legacy_pizero_proj: checkpoints saved before commit 2a901355 ("add
        Npizero") used pi_proj (4-momentum only, no npizero count, no final
        transformer LayerNorm) for the pizero token instead of a dedicated
        pizero_proj. Set True to build the matching (older) architecture when
        loading such a checkpoint -- see load_model_auto in NN_Tools.py, which
        detects this from the checkpoint's state_dict keys automatically.

        polvec_feature_level: 0 (default) leaves the architecture exactly as it
        was, so existing checkpoints load unchanged. 1 adds one extra token per
        hadronic leg carrying the engineered hadronic-current block; 2 adds a
        second token per leg for the "the reconstructed pi0 is spurious"
        alternative. Must match Data.engineered_polvec_features in the config
        that prepared the data, since the columns have to be present in
        input_features.
        """
        super().__init__()
        self.leptonic_mode = leptonic_mode
        self.legacy_pizero_proj = legacy_pizero_proj
        self.polvec_feature_level = int(polvec_feature_level)
        feat_idx = {name: i for i, name in enumerate(input_features)}

        # tau legs that can carry a hadronic current: for leptonic_mode 1 the
        # tau1 leg is leptonic by construction, so only tau2 ever has one.
        self._hcur_legs = ('tau2',) if leptonic_mode == 1 else ('taup', 'taun')
        self._hcur_blocks = ('hcur_', 'hcur_alt_')[:self.polvec_feature_level]
        self._n_hcur_tokens = len(self._hcur_legs) * len(self._hcur_blocks)

        self.met_idx = [feat_idx[f] for f in self._met_feats]

        # projections shared across modes
        self.pi_proj = nn.Linear(4, d_model)   # 4-momentum (pi1, pi2, pi3)
        self.ip_proj = nn.Linear(3, d_model)   # impact parameter / SV 3-vector
        self.met_proj = nn.Linear(2, d_model)
        self.sv_proj = nn.Linear(3, d_model)
        self.lep_proj = nn.Linear(5, d_model)  # lep 4-momentum + ismuon flag
        if not legacy_pizero_proj:
            self.pizero_proj = nn.Linear(5, d_model)  # pizero 4-momentum + npizero count

        if self._n_hcur_tokens:
            self.hcur_proj = nn.Linear(self.HCUR_BLOCK_SIZE, d_model)
            self.hcur_idx = {}
            for block in self._hcur_blocks:
                for leg in self._hcur_legs:
                    names = self._hcur_feats(leg, block)
                    missing = [n for n in names if n not in feat_idx]
                    if missing:
                        raise ValueError(
                            f"polvec_feature_level={self.polvec_feature_level} needs the engineered "
                            f"hadronic-current columns in input_features, but these are missing: "
                            f"{missing}. Prepare the data with Data.engineered_polvec_features "
                            f">= {self.polvec_feature_level} and add the columns to the config.")
                    self.hcur_idx[(block, leg)] = [feat_idx[n] for n in names]

        if leptonic_mode == 0:
            print(">> Using Hadronic Training Embedding")
            self.taup_pi1_idx = [feat_idx[f] for f in self._taup_pi1_feats]
            self.taun_pi1_idx = [feat_idx[f] for f in self._taun_pi1_feats]
            self.taup_charged_ip_idx = [feat_idx[f] for f in self._taup_charged_ip_feats]
            self.taun_charged_ip_idx = [feat_idx[f] for f in self._taun_charged_ip_feats]
            self.taup_pi2_idx = [feat_idx[f] for f in self._taup_pi2_feats]
            self.taun_pi2_idx = [feat_idx[f] for f in self._taun_pi2_feats]
            self.taup_pi3_idx = [feat_idx[f] for f in self._taup_pi3_feats]
            self.taun_pi3_idx = [feat_idx[f] for f in self._taun_pi3_feats]
            self.taup_pizero_idx = [feat_idx[f] for f in self._taup_pizero_feats]
            self.taun_pizero_idx = [feat_idx[f] for f in self._taun_pizero_feats]
            self.taup_npizero_idx = feat_idx['reco_taup_npizero']
            self.taun_npizero_idx = feat_idx['reco_taun_npizero']
            self.taup_sv_idx = [feat_idx[f] for f in self._taup_sv_feats]
            self.taun_sv_idx = [feat_idx[f] for f in self._taun_sv_feats]
            self.taup_haspizero_idx = feat_idx['reco_taup_haspizero']
            self.taun_haspizero_idx = feat_idx['reco_taun_haspizero']
            self.taup_is3prong_idx = feat_idx['reco_taup_is3prong']
            self.taun_is3prong_idx = feat_idx['reco_taun_is3prong']
            self.type_emb = nn.Embedding(13 + self._n_hcur_tokens, d_model)

        elif leptonic_mode == 1:
            print(">> Using SemiLeptonic Training Embedding")
            self.tau1_lep_idx = [feat_idx[f] for f in self._tau1_lep_feats]
            self.tau1_lep_ip_idx = [feat_idx[f] for f in self._tau1_lep_ip_feats]
            self.tau1_ismuon_idx = feat_idx['reco_tau1_ismuon']
            self.tau2_pi1_idx = [feat_idx[f] for f in self._tau2_pi1_feats]
            self.tau2_charged_ip_idx = [feat_idx[f] for f in self._tau2_charged_ip_feats]
            self.tau2_pi2_idx = [feat_idx[f] for f in self._tau2_pi2_feats]
            self.tau2_pi3_idx = [feat_idx[f] for f in self._tau2_pi3_feats]
            self.tau2_pizero_idx = [feat_idx[f] for f in self._tau2_pizero_feats]
            self.tau2_npizero_idx = feat_idx['reco_tau2_npizero']
            self.tau2_sv_idx = [feat_idx[f] for f in self._tau2_sv_feats]
            self.tau2_haspizero_idx = feat_idx['reco_tau2_haspizero']
            self.tau2_is3prong_idx = feat_idx['reco_tau2_is3prong']
            self.type_emb = nn.Embedding(9 + self._n_hcur_tokens, d_model)

        elif leptonic_mode == -1:
            print(">> Using Mixed Hadronic/Leptonic Training Embedding")
            self.taup_pi1_idx = [feat_idx[f] for f in self._taup_pi1_feats]
            self.taun_pi1_idx = [feat_idx[f] for f in self._taun_pi1_feats]
            self.taup_charged_ip_idx = [feat_idx[f] for f in self._taup_charged_ip_feats]
            self.taun_charged_ip_idx = [feat_idx[f] for f in self._taun_charged_ip_feats]
            self.taup_pi2_idx = [feat_idx[f] for f in self._taup_pi2_feats]
            self.taun_pi2_idx = [feat_idx[f] for f in self._taun_pi2_feats]
            self.taup_pi3_idx = [feat_idx[f] for f in self._taup_pi3_feats]
            self.taun_pi3_idx = [feat_idx[f] for f in self._taun_pi3_feats]
            self.taup_pizero_idx = [feat_idx[f] for f in self._taup_pizero_feats]
            self.taun_pizero_idx = [feat_idx[f] for f in self._taun_pizero_feats]
            self.taup_npizero_idx = feat_idx['reco_taup_npizero']
            self.taun_npizero_idx = feat_idx['reco_taun_npizero']
            self.taup_sv_idx = [feat_idx[f] for f in self._taup_sv_feats]
            self.taun_sv_idx = [feat_idx[f] for f in self._taun_sv_feats]
            self.taup_haspizero_idx = feat_idx['reco_taup_haspizero']
            self.taun_haspizero_idx = feat_idx['reco_taun_haspizero']
            self.taup_is3prong_idx = feat_idx['reco_taup_is3prong']
            self.taun_is3prong_idx = feat_idx['reco_taun_is3prong']
            self.taup_ishadronic_idx = feat_idx['reco_taup_ishadronic']
            self.taun_ishadronic_idx = feat_idx['reco_taun_ishadronic']

            self.taup_lep_idx = [feat_idx[f] for f in self._taup_lep_feats]
            self.taun_lep_idx = [feat_idx[f] for f in self._taun_lep_feats]
            self.taup_lep_ip_idx = [feat_idx[f] for f in self._taup_lep_ip_feats]
            self.taun_lep_ip_idx = [feat_idx[f] for f in self._taun_lep_ip_feats]
            self.taup_ismuon_idx = feat_idx['reco_taup_ismuon']
            self.taun_ismuon_idx = feat_idx['reco_taun_ismuon']
            self.type_emb = nn.Embedding(17 + self._n_hcur_tokens, d_model)
        else:
            raise ValueError(f"Unsupported leptonic_mode: {leptonic_mode}")

        if self._n_hcur_tokens:
            self._hcur_flag_idx = {}
            for leg in self._hcur_legs:
                flags = {'is3prong': getattr(self, f'{leg}_is3prong_idx'),
                         'haspizero': getattr(self, f'{leg}_haspizero_idx')}
                if hasattr(self, f'{leg}_ishadronic_idx'):
                    flags['ishadronic'] = getattr(self, f'{leg}_ishadronic_idx')
                self._hcur_flag_idx[leg] = flags

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True,
        )
        if legacy_pizero_proj:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.output_proj = nn.Linear(d_model, context_dim)

    def _pizero_token(self, x, pizero_idx, npizero_idx):
        """New checkpoints: pizero_proj(4-momentum + npizero count).
        legacy_pizero_proj checkpoints: pi_proj(4-momentum only)."""
        if self.legacy_pizero_proj:
            return self.pi_proj(x[:, pizero_idx])
        pizero_input = torch.cat([x[:, pizero_idx], x[:, npizero_idx].unsqueeze(-1)], dim=-1)
        return self.pizero_proj(pizero_input)

    def forward(self, x):
        B, device = x.shape[0], x.device

        if self.leptonic_mode == 0:
            type_embs = self.type_emb(torch.arange(13 + self._n_hcur_tokens, device=device))
            tokens = torch.stack([
                self.pi_proj(x[:, self.taup_pi1_idx]) + type_embs[0],
                self.pi_proj(x[:, self.taun_pi1_idx]) + type_embs[1],
                self.ip_proj(x[:, self.taup_charged_ip_idx]) + type_embs[2],
                self.ip_proj(x[:, self.taun_charged_ip_idx]) + type_embs[3],
                self.pi_proj(x[:, self.taup_pi2_idx]) + type_embs[4],
                self.pi_proj(x[:, self.taun_pi2_idx]) + type_embs[5],
                self.pi_proj(x[:, self.taup_pi3_idx]) + type_embs[6],
                self.pi_proj(x[:, self.taun_pi3_idx]) + type_embs[7],
                self._pizero_token(x, self.taup_pizero_idx, self.taup_npizero_idx) + type_embs[8],
                self._pizero_token(x, self.taun_pizero_idx, self.taun_npizero_idx) + type_embs[9],
                self.met_proj(x[:, self.met_idx]) + type_embs[10],
                self.sv_proj(x[:, self.taup_sv_idx]) + type_embs[11],
                self.sv_proj(x[:, self.taun_sv_idx]) + type_embs[12],
            ], dim=1)

            # Mask undefined columns (DM dependent, True = ignore token)
            pad_mask = torch.zeros(B, 13, dtype=torch.bool, device=device)
            pad_mask[:, 4] = ~x[:, self.taup_is3prong_idx].bool()  # taup pi2
            pad_mask[:, 5] = ~x[:, self.taun_is3prong_idx].bool()  # taun pi2
            pad_mask[:, 6] = ~x[:, self.taup_is3prong_idx].bool()  # taup pi3
            pad_mask[:, 7] = ~x[:, self.taun_is3prong_idx].bool()  # taun pi3
            pad_mask[:, 8] = ~x[:, self.taup_haspizero_idx].bool()  # taup pi0
            pad_mask[:, 9] = ~x[:, self.taun_haspizero_idx].bool()  # taun pi0
            pad_mask[:, 11] = ~x[:, self.taup_is3prong_idx].bool()  # taup SV
            pad_mask[:, 12] = ~x[:, self.taun_is3prong_idx].bool()  # taun SV

        elif self.leptonic_mode == 1:
            type_embs = self.type_emb(torch.arange(9 + self._n_hcur_tokens, device=device))
            lep_input = torch.cat(
                [x[:, self.tau1_lep_idx], x[:, self.tau1_ismuon_idx].unsqueeze(-1)], dim=-1
            )
            tokens = torch.stack([
                self.lep_proj(lep_input) + type_embs[0],  # tau1 lepton
                self.ip_proj(x[:, self.tau1_lep_ip_idx]) + type_embs[1], # tau1 lep IP
                self.pi_proj(x[:, self.tau2_pi1_idx]) + type_embs[2], # tau2 pi1
                self.ip_proj(x[:, self.tau2_charged_ip_idx]) + type_embs[3], # tau2 IP
                self.pi_proj(x[:, self.tau2_pi2_idx]) + type_embs[4],  # tau2 pi2
                self.pi_proj(x[:, self.tau2_pi3_idx]) + type_embs[5],  # tau2 pi3
                self._pizero_token(x, self.tau2_pizero_idx, self.tau2_npizero_idx) + type_embs[6],  # tau2 pi0
                self.met_proj(x[:, self.met_idx]) + type_embs[7],  # MET
                self.sv_proj(x[:, self.tau2_sv_idx]) + type_embs[8],  # tau2 SV
            ], dim=1)

            pad_mask = torch.zeros(B, 9, dtype=torch.bool, device=device)
            pad_mask[:, 4] = ~x[:, self.tau2_is3prong_idx].bool()  # tau2 pi2
            pad_mask[:, 5] = ~x[:, self.tau2_is3prong_idx].bool() # tau2 pi3
            pad_mask[:, 6] = ~x[:, self.tau2_haspizero_idx].bool()  # tau2 pi0
            pad_mask[:, 8] = ~x[:, self.tau2_is3prong_idx].bool()  # tau2 SV

        elif self.leptonic_mode == -1:
            type_embs = self.type_emb(torch.arange(17 + self._n_hcur_tokens, device=device))
            taup_had = x[:, self.taup_ishadronic_idx].bool()
            taun_had = x[:, self.taun_ishadronic_idx].bool()

            taup_lep_input = torch.cat(
                [x[:, self.taup_lep_idx], x[:, self.taup_ismuon_idx].unsqueeze(-1)], dim=-1
            )
            taun_lep_input = torch.cat(
                [x[:, self.taun_lep_idx], x[:, self.taun_ismuon_idx].unsqueeze(-1)], dim=-1
            )

            tokens = torch.stack([
                self.pi_proj(x[:, self.taup_pi1_idx]) + type_embs[0],
                self.pi_proj(x[:, self.taun_pi1_idx]) + type_embs[1],
                self.ip_proj(x[:, self.taup_charged_ip_idx]) + type_embs[2],
                self.ip_proj(x[:, self.taun_charged_ip_idx]) + type_embs[3],
                self.pi_proj(x[:, self.taup_pi2_idx]) + type_embs[4],
                self.pi_proj(x[:, self.taun_pi2_idx]) + type_embs[5],
                self.pi_proj(x[:, self.taup_pi3_idx]) + type_embs[6],
                self.pi_proj(x[:, self.taun_pi3_idx]) + type_embs[7],
                self._pizero_token(x, self.taup_pizero_idx, self.taup_npizero_idx) + type_embs[8],
                self._pizero_token(x, self.taun_pizero_idx, self.taun_npizero_idx) + type_embs[9],
                self.met_proj(x[:, self.met_idx]) + type_embs[10],
                self.sv_proj(x[:, self.taup_sv_idx]) + type_embs[11],
                self.sv_proj(x[:, self.taun_sv_idx]) + type_embs[12],
                self.lep_proj(taup_lep_input) + type_embs[13],
                self.lep_proj(taun_lep_input) + type_embs[14],
                self.ip_proj(x[:, self.taup_lep_ip_idx]) + type_embs[15],
                self.ip_proj(x[:, self.taun_lep_ip_idx]) + type_embs[16],
            ], dim=1)

            # Mask undefined columns. True = ignore token. Hadronic-only tokens
            # (pi1/pi2/pi3/pi0/sv/charged_ip) are masked per-tau when that tau
            # is leptonic; lep/lep_ip tokens are masked per-tau when hadronic.
            pad_mask = torch.zeros(B, 17, dtype=torch.bool, device=device)
            pad_mask[:, 0] = ~taup_had  # taup pi1
            pad_mask[:, 1] = ~taun_had  # taun pi1
            pad_mask[:, 2] = ~taup_had  # taup charged IP
            pad_mask[:, 3] = ~taun_had  # taun charged IP
            pad_mask[:, 4] = ~taup_had | ~x[:, self.taup_is3prong_idx].bool()  # taup pi2
            pad_mask[:, 5] = ~taun_had | ~x[:, self.taun_is3prong_idx].bool()  # taun pi2
            pad_mask[:, 6] = ~taup_had | ~x[:, self.taup_is3prong_idx].bool()  # taup pi3
            pad_mask[:, 7] = ~taun_had | ~x[:, self.taun_is3prong_idx].bool()  # taun pi3
            pad_mask[:, 8] = ~taup_had | ~x[:, self.taup_haspizero_idx].bool()  # taup pi0
            pad_mask[:, 9] = ~taun_had | ~x[:, self.taun_haspizero_idx].bool()  # taun pi0
            pad_mask[:, 11] = ~taup_had | ~x[:, self.taup_is3prong_idx].bool()  # taup SV
            pad_mask[:, 12] = ~taun_had | ~x[:, self.taun_is3prong_idx].bool()  # taun SV
            pad_mask[:, 13] = taup_had  # taup lep
            pad_mask[:, 14] = taun_had  # taun lep
            pad_mask[:, 15] = taup_had  # taup lep IP
            pad_mask[:, 16] = taun_had  # taun lep IP

        if self._n_hcur_tokens:
            tokens, pad_mask = self._append_hcur_tokens(x, tokens, pad_mask, type_embs)

        out = self.transformer(tokens, src_key_padding_mask=pad_mask)

        # mean pool over present tokens only
        present = (~pad_mask).float().unsqueeze(-1)
        context = (out * present).sum(dim=1) / present.sum(dim=1)
        return self.output_proj(context)

    def _append_hcur_tokens(self, x, tokens, pad_mask, type_embs):
        """Append the engineered hadronic-current tokens (see
        engineered_polvec_features.md). Their type embeddings occupy the slots
        after the mode's own tokens, which is why type_emb is sized with
        _n_hcur_tokens added.

        Masking matters here as much as the values do. A DM0 leg has no hadronic
        current at all -- 10 of the 11 numbers are identically zero -- so an
        unmasked token would be attended to as though it were a real particle
        sitting at the origin. The alternative ('the reconstructed pi0 is
        spurious') block is only ever filled for DM11, since a leg reconstructed
        without a pi0 has no pi0 four-vector to drop, so it is masked everywhere
        else.
        """
        B, device = x.shape[0], x.device
        base = tokens.shape[1]
        new_tokens, new_masks = [], []
        slot = base
        for block in self._hcur_blocks:
            for leg in self._hcur_legs:
                new_tokens.append(self.hcur_proj(x[:, self.hcur_idx[(block, leg)]]) + type_embs[slot])
                flags = self._hcur_flag_idx[leg]
                is3 = x[:, flags['is3prong']].bool()
                haspi0 = x[:, flags['haspizero']].bool()
                hadronic = (x[:, flags['ishadronic']].bool() if 'ishadronic' in flags
                            else torch.ones(B, dtype=torch.bool, device=device))
                if block == 'hcur_':
                    present = hadronic & (is3 | haspi0)     # everything except DM0/leptonic
                else:
                    present = hadronic & is3 & haspi0       # DM11 only
                new_masks.append(~present)
                slot += 1
        tokens = torch.cat([tokens, torch.stack(new_tokens, dim=1)], dim=1)
        pad_mask = torch.cat([pad_mask, torch.stack(new_masks, dim=1)], dim=1)
        return tokens, pad_mask

class TransformerRegressor(nn.Module):
    """Transformer encoder (same tokenisation as ConditionalFlow w/ use_transformer=True)
    followed by a small MLP head for direct regression. Trained with MSE loss."""

    def __init__(self, input_features, leptonic_mode, output_dim,
                 context_dim=256, d_model=128, nhead=4, num_layers=3, dropout=0.0):
        super().__init__()
        print("!! INFO: TransformerRegressor — transformer encoder + MSE regression head (no normalizing flow)")
        self.encoder = ParticleTransformerCondition(
            input_features=input_features,
            leptonic_mode=leptonic_mode,
            context_dim=context_dim,
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, output_dim),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class ConditionalFlow(nn.Module):
    def __init__(self,
                 input_dim,
                 raw_condition_dim=None,
                 context_dim=256,
                 cond_hidden_dim=64,
                 cond_num_blocks=2,
                 batch_norm=False,
                 activation=nn.ReLU(),
                 use_transformer=False,
                 input_features=None,
                 leptonic_mode=0,
                 d_model=64,
                 nhead=4,
                 num_transformer_layers=3,
                 dropout=0.0,
                 legacy_pizero_proj=False,
                 polvec_feature_level=0,
                 **flow_kwargs
    ):
        super().__init__()

        if use_transformer:
            print("!! INFO: Using Transformer for conditioning")
            self.condition_net = ParticleTransformerCondition(
                input_features=input_features,
                leptonic_mode=leptonic_mode,
                context_dim=context_dim, d_model=d_model,
                nhead=nhead, num_layers=num_transformer_layers, dropout=dropout,
                legacy_pizero_proj=legacy_pizero_proj,
                polvec_feature_level=polvec_feature_level,
            )
        elif cond_num_blocks == 0:
            print("!! INFO: No conditioning network")
            self.condition_net = nn.Identity()
            context_dim = raw_condition_dim
        else:
            print("!! INFO: Using ResidualNet for conditioning")
            self.condition_net = nets.ResidualNet(
                in_features=raw_condition_dim,
                out_features=context_dim,
                hidden_features=cond_hidden_dim,
                num_blocks=cond_num_blocks,
                activation=activation,
                use_batch_norm=batch_norm
            )

        self.flow = NormalizingFlow(
            input_size=input_dim,
            context_features=context_dim,
            activation=activation,
            **flow_kwargs
        )

    def log_prob(self, inputs, context):
        cond_embed = self.condition_net(context)
        log_prob = self.flow.log_prob(inputs=inputs, context=cond_embed)
        return log_prob

    def sample(self, num_samples, context):
        cond_embed = self.condition_net(context)
        return self.flow.sample(num_samples=num_samples, context=cond_embed)

    def encode(self, inputs, context):
        """
        Map data space -> latent space (x -> z)
        """
        cond_embed = self.condition_net(context)
        z, logabsdet = self.flow._transform.forward(inputs, context=cond_embed)
        return z, logabsdet

    def decode(self, z, context):
        """
        Map latent space -> data space (z -> x)
        """
        cond_embed = self.condition_net(context)
        x, logabsdet = self.flow._transform.inverse(z, context=cond_embed)
        return x, logabsdet

    def sample_and_log_prob(self, num_samples, context):
        """
        Sample and compute log-prob in a single forward pass.
        Returns samples [B, num_samples, F] and log_prob [B, num_samples].
        More efficient than separate sample() + log_prob() calls.
        """
        cond_embed = self.condition_net(context)
        return self.flow.sample_and_log_prob(num_samples=num_samples, context=cond_embed)
