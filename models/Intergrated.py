import torch
from models.networks import MotionAE, LatentDiscriminator, SkeletonEncoder
import os
from utils.utils import ForwardKinematics, build_edge_topology

# package for retargeting between characters in Mixamo
from models.Kinematics import ForwardKinematics as ForwardKinematics_mixamo
from data_preprocess.Mixamo.bvh_parser import BVH_file
from parser.parser_mixamo import get_std_bvh
import torch.nn as nn
from collections import OrderedDict


class IntegratedModel:
    def __init__(self, args, body_parts, njoints, parents, n_topology, device, **kwargs):
        self.args = args
        self.body_parts = body_parts
        self.part_num = len(self.body_parts)
        self.njoints = njoints
        self.indices = [0]

        for part in body_parts:
            self.indices += part
        if args.with_end:
            self.indices_withend = []

            for idx in self.indices:
                self.indices_withend.append(kwargs["not_end"][idx])
            self.indices_withend.extend(kwargs["part_end"])

        else:
            self.indices_withend = self.indices

        if args.with_end:
            self.fk = ForwardKinematics(kwargs["parents_withend"],
                                        kwargs["njoints_withend"],
                                        site_index=kwargs["not_end"])
        else:
            self.fk = ForwardKinematics(parents, njoints)

        self.ae = MotionAE(args, body_parts, njoints).to(device)
        self.skel_enc = SkeletonEncoder(args, body_parts, njoints).to(device)

        if self.args.dis:
            if self.args.dis_mode == 'norm_rotation' or self.args.dis_mode == 'denorm_rotation':
                input_dim = self.args.conv_input * self.njoints + 3
                hidden_dim = self.args.dis_hidden
                self.discriminator = \
                    LatentDiscriminator(args.dis_layers, args.dis_kernel_size,
                                        input_dim, hidden_dim).to(device)
            elif self.args.dis_mode == 'denorm_pos':
                if args.with_end:
                    input_dim = 3 * kwargs["njoints_withend"]
                else:
                    input_dim = 3 * self.njoints
                hidden_dim = self.args.dis_hidden
                self.discriminator = \
                    LatentDiscriminator(args.dis_layers, args.dis_kernel_size,
                                        input_dim, hidden_dim).to(device)


    def parameters(self):
        return self.G_parameters() + self.D_parameters()

    def G_parameters(self):
        parameters = list(self.ae.parameters()) + list(self.skel_enc.parameters())
        return parameters

    def D_parameters(self):
        return list(self.discriminator.parameters())

    def save(self, path, epoch):
        from parser.base import try_mkdir

        path = os.path.join(path, str(epoch))
        try_mkdir(path)

        torch.save(self.ae.state_dict(), os.path.join(path, 'ae.pth'))
        torch.save(self.skel_enc.state_dict(), os.path.join(path, 'skel_enc.pth'))

        if self.args.dis:
            torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))

        print('Save at {} succeed!'.format(path))

    def load(self, path, epoch=None):
        print('loading from', path)
        if not os.path.exists(path):
            raise Exception('Unknown loading path')

        if epoch is None:
            all = [int(q) for q in os.listdir(path) if os.path.isdir(os.path.join(path, q))]
            if len(all) == 0:
                raise Exception('Empty loading path')
            epoch = sorted(all)[-1]

        path = os.path.join(path, str(epoch))
        print('loading from epoch {}......'.format(epoch))

        self.ae.load_state_dict(torch.load(os.path.join(path, 'ae.pth')
                                                     ))
        self.skel_enc.load_state_dict(torch.load(os.path.join(path, 'skel_enc.pth')
                                                     ))

        if os.path.exists(os.path.join(path, 'discriminator.pth')):
            self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator.pth')))
        print('load succeed!')

    def train(self):
        self.ae = self.ae.train()
        self.skel_enc = self.skel_enc.train()
        if self.args.dis:
            self.discriminator = self.discriminator.train()

    def eval(self):
        self.ae = self.ae.eval()
        self.skel_enc = self.skel_enc.eval()
        if self.args.dis:
            self.discriminator = self.discriminator.eval()


class IntegratedModel_Mixamo:
    def __init__(self, args, joint_topology, device, characters):
        self.args = args
        self.joint_topology = joint_topology
        self.edges = build_edge_topology(joint_topology, torch.zeros((len(joint_topology), 3)))
        self.fk = ForwardKinematics_mixamo(args, self.edges)

        self.height = []
        self.real_height = []
        for char in characters:
            if args.use_sep_ee:
                h = BVH_file(get_std_bvh(dataset=char)).get_ee_length()
            else:
                h = BVH_file(get_std_bvh(dataset=char)).get_height()
            if args.ee_loss_fact == 'learn':
                h = torch.tensor(h, dtype=torch.float)
            else:
                h = torch.tensor(h, dtype=torch.float, requires_grad=False)
            self.real_height.append(BVH_file(get_std_bvh(dataset=char)).get_height())
            self.height.append(h.unsqueeze(0))
        self.real_height = torch.tensor(self.real_height, device=device)
        self.height = torch.cat(self.height, dim=0)
        self.height = self.height.to(device)
        if not args.use_sep_ee: self.height.unsqueeze_(-1)
        if args.ee_loss_fact == 'learn': self.height_para = [self.height]
        else: self.height_para = []

        if args.model == "pan":
            self.auto_encoder = MotionAE(args, self.edges, None).to(device)
            self.static_encoder = SkeletonEncoder(args, self.edges, None).to(device)
            self.discriminator = LatentDiscriminator(3, 15, (len(self.edges) + 1)*3, 256, is_lafan1=False)


    def parameters(self):
        return self.G_parameters() + self.D_parameters()

    def G_parameters(self):
        return list(self.auto_encoder.parameters()) + list(self.static_encoder.parameters()) + self.height_para

    def D_parameters(self):
        return list(self.discriminator.parameters())

    def train(self):
        self.auto_encoder.train()
        self.discriminator.train()
        self.static_encoder.train()

    def eval(self):
        self.auto_encoder.eval()
        self.discriminator.eval()
        self.static_encoder.eval()

    def save(self, path, epoch):
        from parser.parser_mixamo import try_mkdir

        path = os.path.join(path, str(epoch))
        try_mkdir(path)

        torch.save(self.height, os.path.join(path, 'height.pt'))
        torch.save(self.auto_encoder.state_dict(), os.path.join(path, 'auto_encoder.pt'))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pt'))
        torch.save(self.static_encoder.state_dict(), os.path.join(path, 'static_encoder.pt'))

        print('Save at {} succeed!'.format(path))

    def load(self, path, epoch=None):
        print('loading from', path)
        if not os.path.exists(path):
            raise Exception('Unknown loading path')

        if epoch is None:
            all = [int(q) for q in os.listdir(path) if os.path.isdir(os.path.join(path, q))]
            if len(all) == 0:
                raise Exception('Empty loading path')
            epoch = sorted(all)[-1]

        path = os.path.join(path, str(epoch))
        print('loading from epoch {}......'.format(epoch))
        if self.args.use_parallel:
            self.load_network(self.auto_encoder, os.path.join(path, 'auto_encoder.pt'))
            self.load_network(self.static_encoder, os.path.join(path, 'static_encoder.pt'))
        else:
            self.auto_encoder.load_state_dict(torch.load(os.path.join(path, 'auto_encoder.pt'),
                                                         map_location=self.args.cuda_device))
            self.static_encoder.load_state_dict(torch.load(os.path.join(path, 'static_encoder.pt'),
                                                           map_location=self.args.cuda_device))

        print('load succeed!')

    def load_network(self, network, save_path):
        state_dict = torch.load(save_path)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            namekey = k[7:]  # remove `module.`
            new_state_dict[namekey] = v
        # load params
        network.load_state_dict(new_state_dict)
        return network

    def DataParallel(self):
        self.static_encoder = nn.DataParallel(self.static_encoder)
        self.auto_encoder = nn.DataParallel(self.auto_encoder)
        self.discriminator = nn.DataParallel(self.discriminator)
