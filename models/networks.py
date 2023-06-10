import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.multi_attention_forward import multi_head_attention_forward
# get body parts in Lafan1 and dog sets
from utils.utils import get_part_matrix, get_transformer_matrix, get_offset_part_matrix
# get body parts in Mixamo set
from utils.utils import getbodyparts, calselfmask


class MotionAE(nn.Module):
    def __init__(self, args, correspondence, njoints):
        super(MotionAE, self).__init__()
        self.nparts = len(correspondence)
        self.enc = MotionEncoder(args, correspondence, njoints)
        self.dec = MotionDecoder(args, self.enc)

    def forward(self, input, offset=None):
        latent = self.enc(input)
        result = self.dec(latent, offset)
        return latent, result

    def outformat2input(self, output):
        """
        :param output: the decoder output
        :return: transposed output
        """
        # output shape : B T C
        # input shape B C T
        return output.transpose(1, 2)


class MotionEncoder(nn.Module):

    def __init__(self, args, correspondence, njoints, activation="gelu"):
        super(MotionEncoder, self).__init__()

        self.is_lafan1 = True   # This is for quadruped and biped retargeting
        if njoints is None:
            self.is_lafan1 = False # This is for retargeting between Mixamo characters

        if self.is_lafan1:
            self.args = args
            self.correspondence = correspondence
            self.nparts = len(correspondence)
            self.njoints = njoints + 1  # append root volocity
        else:
            body_parts = getbodyparts(correspondence)
            self.body_parts = body_parts
            self.nparts = len(body_parts)
            self.njoints = len(correspondence) + 1  # append root volocity


        self.num_layers = args.transformer_layers
        self.latent_dim = args.transformer_latents
        self.ff_size = args.transformer_ffsize
        self.num_heads = args.transformer_heads
        self.dropout = args.transformer_dropout
        self.src_dim = args.transformer_srcdim
        self.activation = activation
        self.act_f = nn.LeakyReLU(negative_slope=0.2)

        device = torch.device("cuda:0" if (torch.cuda.is_available()) else 'cpu')

        if self.is_lafan1:
            self.attention_mask = get_transformer_matrix(correspondence, self.njoints).to(device)
            self.mask = get_part_matrix(correspondence, self.njoints)
        else:
            self.attention_mask = calselfmask(body_parts, self.njoints, correspondence, is_conv=False).to(device)
            self.mask = calselfmask(body_parts, self.njoints, correspondence, is_conv=True).to(device)

        # Spatial Transformer (frame by frame)
        self.joint_pos_encoder = PositionalEncoding(self.src_dim, self.latent_dim, self.dropout)

        spaceTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)

        self.spatialTransEncoder = nn.TransformerEncoder(spaceTransEncoderLayer,
                                                         num_layers=self.num_layers)

        self.parameter_part = nn.Parameter(torch.randn(self.nparts, 1, self.latent_dim) * 0.1)

        #### Temporal compression, conv1d.
        self.conv_input_dim = args.transformer_latents * self.nparts
        self.raw_input_dim = int(self.njoints * self.src_dim)
        padding = (args.kernel_size - 1) // 2
        if args.padding_mode == 'reflection' or args.padding_mode == 'reflect':
            padding_mode = 'reflect'
        else:
            padding_mode = args.padding_mode

        self.layers = nn.ModuleList()

        # add residual connection
        self.conv_residual = BodyPartConv(self.nparts, self.mask, self.raw_input_dim,
                                                   self.conv_input_dim, args.kernel_size, self.njoints,
                                                   stride=2, padding=padding, bias=True,
                                                   padding_mode=padding_mode, first_layer=True)

        # temporal layers
        self.conv1 = BodyPartConv(self.nparts, self.mask, self.conv_input_dim,
                                                   self.conv_input_dim, args.kernel_size, self.njoints,
                                                   stride=2, padding=padding, bias=True,
                                                   padding_mode=padding_mode)

        self.conv2 = BodyPartConv(self.nparts, self.mask, self.conv_input_dim,
                                                   self.conv_input_dim, args.kernel_size, self.njoints,
                                                   stride=2, padding=padding, bias=True,
                                                   padding_mode=padding_mode)

    def forward(self, x):
        if not self.is_lafan1:
            x = torch.cat((x, torch.zeros_like(x[:, [0], :])), dim=1)  # B C T

        raw_x = x.clone()
        # spatial modeling by pose-aware attention and body parts
        b_size, c_size, t_size = x.shape[0], x.shape[1], x.shape[2]
        j_num = c_size // self.src_dim
        x = x.reshape(b_size, j_num, self.src_dim, t_size).transpose(0, 1).transpose(2, 3) \
            .reshape(j_num, b_size * t_size, self.src_dim)  # J BT E
        encoding = self.joint_pos_encoder(x)
        encoding_app = torch.cat((self.parameter_part.repeat(1, encoding.shape[1], 1), encoding), dim=0)
        final = self.spatialTransEncoder(encoding_app, mask=self.attention_mask)
        final_parts = final[:self.nparts].transpose(0, 1).reshape(b_size, t_size, -1).transpose(1, 2)  # B * nparts E * T
        final_parts = final_parts.reshape(b_size, -1, t_size)

        # temporal compression
        residual_connection = self.conv_residual(raw_x)
        final_parts = self.conv1(final_parts)
        final_parts = self.act_f(residual_connection + final_parts)
        final_parts = self.act_f(self.conv2(final_parts))

        return final_parts


class MotionDecoder(nn.Module):
    def __init__(self, args, enc: MotionEncoder):
        super(MotionDecoder, self).__init__()

        self.is_lafan1 = enc.is_lafan1 # True for biped-quadruped retargeting

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2
        if args.padding_mode == 'reflection' or args.padding_mode == 'reflect':
            padding_mode = 'reflect'
        else:
            padding_mode = args.padding_mode

        self.args = args
        self.num_layers = args.conv_layers

        self.layers = nn.ModuleList()
        in_channels = enc.conv_input_dim

        # Decovolutional and Upsampling layers
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                last_layer = True
                out_channels = enc.raw_input_dim
            else:
                last_layer = False
                out_channels = enc.conv_input_dim

            layer_components = []

            layer_components.append(nn.Upsample(scale_factor=2, mode=args.upsampling, align_corners=False))
            layer_components.append(BodyPartConv(enc.nparts, enc.mask, in_channels,
                                         out_channels, kernel_size, enc.njoints, stride=1, padding=padding,
                                         padding_mode=padding_mode, bias=True, last_layer=last_layer))
            if i != self.num_layers - 1:
                layer_components.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*layer_components))

            if args.skeleton_info == 'additive':
                self.add_offset = True
            else:
                self.add_offset = False

    def forward(self, x, offset=None):

        if self.add_offset and offset is not None:  # fuse motion and skeleton codes in an additiv
            x = x + offset / 100
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if self.is_lafan1:
            return x.transpose(1, 2)
        else:
            x = x[:, :-1, :]
            return x


class SkeletonEncoder(nn.Module):
    def __init__(self, args, correspondence, njoints):
        super(SkeletonEncoder, self).__init__()
        self.args = args
        self.is_lafan1 = True
        if njoints is None:
            self.is_lafan1 = False
        if self.is_lafan1:    # Skeleton Encoder for biped and quadruped retargeting
            self.correspondence = correspondence
            self.nparts = len(correspondence)
            num_offsets = njoints - 1  # remove the root joint translation
            input_dim = num_offsets * 3  # offsets dimension
            output_dim = self.nparts * args.dim_per_part
            self.mask = get_offset_part_matrix(correspondence, num_offsets)
        else:
            # Skeleton Encoder for retargeting between characters in Mixamo datasets
            self.topology = correspondence
            self.njoints = len(self.topology) + 1
            num_offsets = self.njoints
            body_parts = getbodyparts(self.topology)
            self.nparts = len(body_parts)
            input_dim = self.njoints * 3  # offsets dimension
            output_dim = self.nparts * args.transformer_latents
            self.mask = calselfmask(body_parts, self.njoints, self.topology, is_conv=True)

        # Body part related MLP
        self.linear1 = BodyPartMlp(self.nparts, self.mask, input_dim,
                       output_dim, num_offsets, first_layer=True)
        self.linear2 = BodyPartMlp(self.nparts, self.mask, output_dim,
                       output_dim, num_offsets)
        self.linear3 = BodyPartMlp(self.nparts, self.mask, output_dim,
                       output_dim, num_offsets)
        self.act_f = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, offsets):
        if not self.is_lafan1:
            offsets = offsets.reshape(offsets.shape[0], -1)
        deep_offsets = self.act_f(self.linear1(offsets))
        deep_offsets = self.act_f(self.linear2(deep_offsets))
        deep_offsets = self.act_f(self.linear3(deep_offsets))
        if not self.is_lafan1:
            return deep_offsets.unsqueeze(-1)
        else:
            return deep_offsets


class LatentDiscriminator(nn.Module):
    def __init__(self, num_layers, kernel_size, input_dim, hidden_dim, out_dim=1, is_lafan1=True):
        super(LatentDiscriminator, self).__init__()
        self.layers = nn.ModuleList()
        self.is_lafan1 = is_lafan1
        for i in range(num_layers):
            layer_comp = []
            if i == 0:
                layer_comp.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, bias=True))
                layer_comp.append(nn.LeakyReLU(negative_slope=0.2))
            elif i == num_layers-1:
                layer_comp.append(nn.Conv1d(hidden_dim, out_dim, kernel_size=1, bias=True))
            else:
                layer_comp.append(nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=1, bias=True))
                layer_comp.append(nn.LeakyReLU(negative_slope=0.2))
                hidden_dim = hidden_dim // 2
            self.layers.append(nn.Sequential(*layer_comp))

    def forward(self, x):
        if self.is_lafan1: # for biped-quadruped retargeting
            # x shape: batch_size, channel num, time
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            # for Mixamo dataset
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            for layer in self.layers:
                x = layer(x)
            return torch.sigmoid(x).squeeze()


class BodyPartConv(nn.Module):

    def __init__(self, body_lens, masks, in_channels, out_channels, kernel_size, joint_num, stride=1, padding=0,
                 bias=True, padding_mode='zeros', add_offset=False, first_layer=False, last_layer=False):
        super(BodyPartConv, self).__init__()

        self.in_channels_per_part = in_channels // body_lens
        self.out_channels_per_part = out_channels // body_lens

        if first_layer:

            if in_channels % joint_num != 0 or out_channels % body_lens != 0:
                raise Exception('BAD')

        if last_layer:
            if in_channels % body_lens != 0 or out_channels % joint_num != 0:
                raise Exception('BAD')

        if not last_layer and not first_layer:
            if in_channels % body_lens != 0 or out_channels % body_lens != 0:
                raise Exception('BAD')


        if padding_mode == 'zeros': padding_mode = 'constant'
        if padding_mode == 'reflect': padding_mode = 'reflect'

        self.add_offset = add_offset
        self.body_lens = body_lens

        self.stride = stride
        self.dilation = 1
        self.groups = 1
        self.padding = padding
        self.padding_mode = padding_mode
        self._padding_repeated_twice = (padding, padding)

        self.in_neighbour_list = []
        if not first_layer:
            for i in range(self.body_lens):
                expanded = list(np.arange(i*self.in_channels_per_part, (i+1)*self.in_channels_per_part))
                self.in_neighbour_list.append(expanded)
        else:
            per_joint_channels = in_channels // joint_num
            for i in range(masks.shape[0]):
                index = list(torch.where(masks[i] == 1)[0])
                expanded = []
                for k in range(len(index)):
                    for j in range(per_joint_channels):
                        expanded.append(index[k] * per_joint_channels + j)
                self.in_neighbour_list.append(expanded)


        self.out_neighbour_list = []

        if not last_layer:
            for i in range(self.body_lens):
                expanded = list(np.arange(i*self.out_channels_per_part, (i+1)*self.out_channels_per_part))
                self.out_neighbour_list.append(expanded)
        else:
            per_joint_channels = out_channels // joint_num
            for i in range(self.body_lens):
                index = list(torch.where(masks[i] == 1)[0])
                expanded = []
                for k in range(len(index)):
                    for j in range(per_joint_channels):
                        expanded.append(index[k] * per_joint_channels + j)
                self.out_neighbour_list.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels, kernel_size)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.register_parameter('bias', None)

        self.mask = torch.zeros_like(self.weight)
        for i in range(self.body_lens):
            for j in self.out_neighbour_list[i]:
                self.mask[j, self.in_neighbour_list[i], ...] = 1
        if last_layer:
            def check_inpart(i):
                a = False
                for _, out_list in enumerate(self.out_neighbour_list):
                    if i in out_list:
                        a = True
                return a
            for k in range(out_channels):
                if not check_inpart(k):
                    self.mask[k, ...] = 1

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        self.description = 'SkeletonConv(in_channels_per_armature={}, out_channels_per_armature={}, kernel_size={}, ' \
                           'joint_num={}, stride={}, padding={}, bias={})'.format(
            in_channels // joint_num, out_channels // joint_num, kernel_size, joint_num, stride, padding, bias
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.Parameter(self.weight)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight_masked = self.weight * self.mask
        res = F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                       weight_masked, self.bias, self.stride,
                       0, self.dilation, self.groups)
        return res


class BodyPartMlp(nn.Module):

    def __init__(self, body_lens, masks, in_channels, out_channels, joint_num,
                 bias=True, first_layer=False, last_layer=False):

        super(BodyPartMlp, self).__init__()

        self.in_channels_per_part = in_channels // body_lens
        self.out_channels_per_part = out_channels // body_lens

        if first_layer:
            if in_channels % joint_num != 0 or out_channels % body_lens != 0:
                raise Exception('BAD')

        if last_layer:
            if in_channels % body_lens != 0 or out_channels % joint_num != 0:
                raise Exception('BAD')

        if not last_layer and not first_layer:
            if in_channels % body_lens != 0 or out_channels % body_lens != 0:
                raise Exception('BAD')

        self.body_lens = body_lens

        self.in_neighbour_list = []
        if not first_layer:
            for i in range(self.body_lens):
                expanded = list(np.arange(i*self.in_channels_per_part, (i+1)*self.in_channels_per_part))
                self.in_neighbour_list.append(expanded)
        else:
            per_joint_channels = in_channels // joint_num
            for i in range(masks.shape[0]):
                index = list(torch.where(masks[i] == 1)[0])
                expanded = []
                for k in range(len(index)):
                    for j in range(per_joint_channels):
                        expanded.append(index[k] * per_joint_channels + j)
                self.in_neighbour_list.append(expanded)

        self.out_neighbour_list = []

        if not last_layer:
            for i in range(self.body_lens):
                expanded = list(np.arange(i*self.out_channels_per_part, (i+1)*self.out_channels_per_part))
                self.out_neighbour_list.append(expanded)
        else:
            per_joint_channels = out_channels // joint_num
            for i in range(self.body_lens):
                index = list(torch.where(masks[i] == 1)[0])
                expanded = []
                for k in range(len(index)):
                    for j in range(per_joint_channels):
                        expanded.append(index[k] * per_joint_channels + j)
                self.out_neighbour_list.append(expanded)

        self.weight = torch.zeros(out_channels, in_channels)
        if bias:
            self.bias = torch.zeros(out_channels)
        else:
            self.register_parameter('bias', None)

        self.mask = torch.zeros_like(self.weight)
        for i in range(self.body_lens):
            for j in self.out_neighbour_list[i]:
                self.mask[j, self.in_neighbour_list[i], ...] = 1

        if last_layer:
            def check_inpart(i):
                a = False
                for _, out_list in enumerate(self.out_neighbour_list):
                    if i in out_list:
                        a = True
                return a
            for k in range(out_channels):
                if not check_inpart(k):
                    self.mask[k, ...] = 1

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.Parameter(self.weight)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight_masked = self.weight * self.mask
        res = F.linear(input, weight_masked, self.bias)
        return res


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, layer_cache=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, layer_cache=layer_cache)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, layer_cache=layer_cache)


class PositionalEncoding(nn.Module):
    def __init__(self, src_dim, embed_dim, dropout, max_len=100, hid_dim=512):
        """
        :param src_dim:  orignal input dimension
        :param embed_dim: embedding dimension
        :param dropout: dropout rate
        :param max_len: max length
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(src_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, embed_dim)
        self.relu = nn.ReLU()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embed_dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / embed_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, step=None):
        """
        :param input: L x N x D
        :param step:
        :return:
        """
        # raw_shape = input.shape[:-2]
        # j_num, f_dim = input.shape[-2], input.shape[-1]
        # input = input.reshape(-1, j_num, f_dim).transpose(0, 1)
        emb = self.linear2(self.relu(self.linear1(input)))
        emb = emb * math.sqrt(self.embed_dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        # emb = emb.transpose(0, 1).reshape(raw_shape + (j_num, -1))
        return emb


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, npart, dim_feedforward=1024, dropout=0.2, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.npart = npart
        # for n in range(npart):
        #     self.parameter_part.append(nn.Parameter(torch.randn(1, 1, d_model)))
        # self.parameter_part = torch.cat(self.parameter_part, dim=0)
        self.parameter_part = nn.Parameter(torch.randn(npart, 1, d_model))

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        else:
            raise RuntimeError("activation should be relu/gelu, not %s." % activation)

    def forward(self, src, srcapp_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required). L N D
            srcapp_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        b_size, t_size, j_num = src.shape[0], src.shape[1], src.shape[2]
        src = src.reshape(b_size * t_size, j_num, -1).transpose(0, 1)   # J BT E

        src_app = torch.cat((self.parameter_part.repeat(1, src.shape[1], 1), src), dim=0)

        src2, attn = self.self_attn(src_app, src_app, src_app, attn_mask=srcapp_mask,
                                    key_padding_mask=src_key_padding_mask)
        # src = src + self.dropout1(src2)
        src = self.dropout1(src2)
        src = self.norm1(src)

        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        src = src + self.dropout2(src2)
        # src = self.norm2(src)
        src = src[:self.npart, ...]
        src = src.transpose(0, 1).reshape(b_size, t_size, self.npart, -1)
        return src, attn








