from models.Intergrated import IntegratedModel
from models.functions import *
import torch
import os
from parser.base import try_mkdir
from models.base_model import BaseModel
from utils.utils import get_lpos
import torch.nn as nn


class PAN_model(BaseModel):
    def __init__(self, args, body_parts, datasets, topology_name):
        super(PAN_model, self).__init__(args)
        self.D_parameters = []
        self.G_parameters = []
        self.models = []
        self.args = args
        self.datasets = datasets
        self.n_topology = len(body_parts)
        self.topology_name = topology_name

        self.criterion_gan = get_gan_loss(args.dis_loss_type)
        self.criterion_rec = get_rec_loss(args.rec_loss_type)
        self.criterion_root = get_root_loss(args.root_loss_type)
        self.criterion_kine = get_kine_loss(args.global_kine_loss_type)
        self.criterion_cycle = get_cycle_loss(args.cyc_loss_type)
        self.criterion_cycle_latent = get_cycle_latent_loss(args.cyc_latent_loss_type)
        self.criterion_root_v = get_retar_root_v_loss(args.retar_vel_loss_type)
        self.mse = nn.MSELoss()

        for i in range(len(topology_name)):
            if args.with_end:
                if topology_name[i] == 'human':
                    part_end = args.hum_end

                elif topology_name[i] == 'dog':
                    part_end = args.dog_end

                else:
                    raise NotImplementedError

                model = IntegratedModel(args, body_parts[i], datasets[i].njoints,
                                        datasets[i].parents,
                                        len(topology_name), self.device,
                                        parents_withend=datasets[i].parents_withend,
                                        njoints_withend=datasets[i].njoints_withend,
                                        not_end=datasets[i].not_end,
                                        part_end=part_end)
            else:
                model = IntegratedModel(args, body_parts[i], datasets[i].njoints,
                                       datasets[i].parents, len(topology_name), self.device)
            if args.is_train:
                model.train()
            else:
                model.eval()
            self.models.append(model)
            self.D_parameters += model.D_parameters()
            self.G_parameters += model.G_parameters()

        if args.is_train:
            self.fake_pools = []
            self.optimizerD = get_optimizer(args.optimizer, self.D_parameters, lr=args.lr_d)
            self.optimizerG = get_optimizer(args.optimizer, self.G_parameters, lr=args.lr_g)
            self.optimizers = [self.optimizerD, self.optimizerG]

    def discriminator_requires_grad_(self, requires_grad):
        for model in self.models:
            for para in model.discriminator.parameters():
                para.requires_grad = requires_grad

    def set_input(self, input):
        self.motions_input = []
        self.offsets = []
        self.offsets_withend = []
        for i, (motion, offsets, offsets_withend) in enumerate(input):
            self.motions_input.append(motion.float().to(self.device))
            self.offsets.append(offsets.float().to(self.device))
            self.offsets_withend.append(offsets_withend.float().to(self.device))

    def forward(self):
        self.motion = []
        self.motion_denorm = []
        self.skel_rep = []
        self.latents = []
        self.gt_pos = []
        self.gt_local_pos = []
        self.rec = []
        self.rec_denorm = []
        self.rec_pos = []
        self.rec_local_pos = []

        self.cyc = []
        self.cyc_denorm = []
        self.cyc_pos = []
        self.cyc_local_pos = []
        self.cyc_latents = []

        self.fake_retar = []
        self.fake_retar_denorm = []
        self.fake_pos = []
        self.fake_local_pos = []
        self.fake_latents = []
        self.retar_latents = []


        # reconstruct
        for i in range(self.n_topology):
            motion = self.motions_input[i]
            self.skel_rep.append(self.models[i].skel_enc(self.offsets[i]).unsqueeze(-1))
            latent, rec = self.models[i].ae(motion, self.skel_rep[i])
            rec_denorm = self.datasets[i].denorm(rec, transpose=False)

            if self.args.with_end:
                lpos = get_lpos(self.offsets_withend[i], self.args.time_size,
                                self.datasets[i].njoints_withend, self.device)
            else:
                lpos = get_lpos(self.offsets[i], self.args.time_size,
                                self.datasets[i].njoints, self.device)

            rec_pos, rec_local_pos = self.models[i].fk.forward(rec_denorm, lpos)
            motion_denorm = self.datasets[i].denorm(motion, transpose=True)
            gt_pos, gt_local_pos = self.models[i].fk.forward(motion_denorm, lpos)

            self.motion.append(motion)
            self.motion_denorm.append(motion_denorm)
            self.rec.append(rec)
            self.rec_denorm.append(rec_denorm)
            self.rec_pos.append(rec_pos)
            self.latents.append(latent)
            self.gt_pos.append(gt_pos)
            self.gt_local_pos.append(gt_local_pos)
            self.rec_local_pos.append(rec_local_pos)

        # retargeting
        for i in range(self.n_topology):
            a = 0
            if self.args.with_end:
                lpos_i = get_lpos(self.offsets_withend[i], self.args.time_size,
                                self.datasets[i].njoints_withend, self.device)
            else:
                lpos_i = get_lpos(self.offsets[i], self.args.time_size,
                                self.datasets[i].njoints, self.device)
            for j in range(self.n_topology):
                if j == i:
                    continue
                else:
                    retar_latent = get_retar_latents(self, i)
                    fake_retar = self.models[j].ae.dec(retar_latent, self.skel_rep[j])
                    fake_retar_input = self.models[j].ae.outformat2input(fake_retar)
                    fake_latent = self.models[j].ae.enc(fake_retar_input)

                    # cycle
                    cyc_latent = get_cyc_latents(self, fake_latent)
                    cyc = self.models[i].ae.dec(cyc_latent, self.skel_rep[i])
                    cyc_denorm = self.datasets[i].denorm(cyc, transpose=False)
                    cyc_pos, cyc_local_pos = self.models[i].fk.forward(cyc_denorm, lpos_i)
                    fake_retar_denorm = self.datasets[j].denorm(fake_retar, transpose=False)

                    if self.args.with_end:
                        lpos_j = get_lpos(self.offsets_withend[j], self.args.time_size,
                                          self.datasets[j].njoints_withend, self.device)
                    else:
                        lpos_j = get_lpos(self.offsets[j], self.args.time_size,
                                          self.datasets[j].njoints, self.device)
                    fake_pos, fake_local_pos = self.models[j].fk.forward(fake_retar_denorm, lpos_j)

                    self.retar_latents.append(retar_latent)
                    self.fake_retar.append(fake_retar)
                    self.fake_retar_denorm.append(fake_retar_denorm)
                    self.fake_pos.append(fake_pos)
                    self.fake_local_pos.append(fake_local_pos)
                    self.fake_latents.append(fake_latent)

                    self.cyc.append(cyc)
                    self.cyc_latents.append(cyc_latent)
                    self.cyc_denorm.append(cyc_denorm)
                    self.cyc_pos.append(cyc_pos)
                    self.cyc_local_pos.append(cyc_local_pos)

                a += 1

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_D = 0
        """
        A->B, B->A [0, 1]
        """
        p = 0
        for i in range(self.n_topology):
            for j in range(self.n_topology):
                if j == i:
                    continue
                else:
                    true_input = get_discriminator_input(self, self.args.dis_mode, j, True)
                    fake_input = get_discriminator_input(self, self.args.dis_mode, p, False)
                    loss_Ds = self.backward_D_basic(self.models[j].discriminator,
                                                    true_input.detach(), fake_input.detach())
                    self.loss_D += loss_Ds
                    self.loss_recoder.add_scalar('D_loss_{}'.format(i), loss_Ds)
                p += 1

    def backward_G(self):
        # rec_loss and gan loss
        self.rec_losses = []
        self.rec_loss = 0
        self.cycle_loss = 0
        self.loss_G = 0
        self.retar_root_v_loss = 0
        self.loss_G_total = 0

        # reconstruction loss
        for i in range(self.n_topology):
            input_0, input_1 = get_recloss_input(self, self.args.rec_loss_type, i)
            indices = self.models[i].indices
            indices_withend = self.models[i].indices_withend
            # rec_loss1: reconstruct quaternions
            rec_loss1 = self.criterion_rec(input_0,
                                           input_1,
                                           self.datasets[i].njoints,
                                           indices=indices)

            self.loss_recoder.add_scalar('rec_loss_quater_{}'.format(i), rec_loss1)

            input_pos = self.motion[i][:, -4:, :].transpose(1, 2)
            rec_pos = self.rec[i][..., -4:]

            # rec_loss2: add more weigths on root velocity
            rec_loss2 = self.criterion_root(input_pos, rec_pos)
            self.loss_recoder.add_scalar('rec_loss_global_{}'.format(i), rec_loss2)

            # rec_loss3: reconstruct kinematic positions
            rec_loss3 = self.criterion_kine(self.gt_pos[i], self.rec_pos[i], indices=indices_withend)
            self.loss_recoder.add_scalar('rec_loss_position_{}'.format(i), rec_loss3)

            rec_loss = rec_loss1 + rec_loss2 * 100 + rec_loss3 * 1e-2

            self.rec_losses.append(rec_loss)
            self.rec_loss += rec_loss

        p = 0
        for src in range(self.n_topology):

            indices_withend = self.models[src].indices_withend
            for dst in range(self.n_topology):
                src_joints = self.datasets[src].njoints
                dst_joints = self.datasets[dst].njoints
                if dst == src:
                    continue
                else:
                    # cycle consistency loss for joint positions and latent codes
                    cycle_loss = self.criterion_cycle(self, src, p, indices_withend)
                    cycle_latent_loss = self.criterion_cycle_latent(self, src, p)
                    self.loss_recoder.add_scalar('cycle_loss_{}_{}'.format(src, dst), cycle_loss)
                    self.loss_recoder.add_scalar('cycle_latent_loss_{}_{}'.format(src, dst), cycle_latent_loss)
                    self.cycle_loss += cycle_loss
                    self.cycle_loss += cycle_latent_loss

                    src_vector = self.motion_denorm[src][..., src_joints * 4: src_joints * 4 + 3]
                    retar_vector = self.fake_retar_denorm[p][..., dst_joints * 4: dst_joints * 4 + 3]

                    src_min_vel = torch.Tensor(self.datasets[src].min_vel).to(self.device)
                    src_max_vel = torch.Tensor(self.datasets[src].max_vel).to(self.device)
                    dst_min_vel = torch.Tensor(self.datasets[dst].min_vel).to(self.device)
                    dst_max_vel = torch.Tensor(self.datasets[dst].max_vel).to(self.device)
                    input_vel_scalar = (torch.norm(src_vector, dim=-1, p=2, keepdim=True) -
                                 src_min_vel)/(src_max_vel - src_min_vel)
                    retar_vel_scalar = (torch.norm(retar_vector, dim=-1, p=2, keepdim=True) -
                                 dst_min_vel)/(dst_max_vel - dst_min_vel)

                    input_vel = input_vel_scalar * src_vector\
                                /torch.norm(src_vector, dim=-1, p=2, keepdim=True)

                    retar_vel = retar_vel_scalar * retar_vector\
                                /torch.norm(retar_vector, dim=-1, p=2, keepdim=True)

                    if self.args.retar_vel_matching == 'mapping':
                        retar_root_v_loss = self.criterion_root_v(input_vel, retar_vel)
                    elif self.args.retar_vel_matching == 'direct':
                        retar_root_v_loss = self.criterion_root_v(self.motion_denorm[src][..., -4:],
                                                                  self.fake_retar_denorm[p][..., -4:])
                    elif self.args.retar_vel_matching == 'direction':
                        retar_root_v_loss = self.criterion_root_v(src_vector/torch.norm(src_vector,
                                            dim=-1, p=2, keepdim=True),
                                                                  retar_vector/torch.norm(retar_vector,
                                            dim=-1, p=2, keepdim=True))

                    retar_root_v_loss += self.mse(self.motion_denorm[src][..., src_joints * 4 + 3: src_joints * 4 + 4],
                                             self.fake_retar_denorm[p][..., dst_joints * 4 + 3: dst_joints * 4 + 4])
                    self.loss_recoder.add_scalar('retar_root_v_loss_{}_{}'.format(src, dst), retar_root_v_loss)
                    self.retar_root_v_loss += retar_root_v_loss

                    if self.args.dis:
                        dis_input = get_discriminator_input(self, self.args.dis_mode, p, False)
                        loss_G = self.criterion_gan(self.models[dst].discriminator(dis_input), True)
                    else:
                        loss_G = torch.tensor(0)
                    self.loss_recoder.add_scalar('G_loss_{}_{}'.format(src, dst), loss_G)
                    self.loss_G += loss_G

                p += 1

        self.loss_G_total = self.rec_loss * self.args.lambda_rec + \
                            self.cycle_loss * self.args.lambda_cycle + \
                            self.loss_G * 1 + \
                            self.retar_root_v_loss * self.args.lambda_retar_vel
        self.loss_recoder.add_scalar('G_loss_total', self.loss_G_total)
        self.loss_G_total.backward()

    def optimize_parameters(self):
        self.forward()

        # update Gs
        self.discriminator_requires_grad_(False)
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        # update Ds
        if self.args.dis:
            self.discriminator_requires_grad_(True)
            self.optimizerD.zero_grad()
            self.backward_D()
            self.optimizerD.step()
        else:
            self.loss_D = torch.tensor(0)

    def verbose(self):
        res = {'rec_loss_0': self.rec_losses[0].item(),
               'rec_loss_1': self.rec_losses[1].item(),
               'cycle_loss': self.cycle_loss.item(),
               'D_loss_gan': self.loss_D.item(),
               'G_loss_gan': self.loss_G.item()}
        return sorted(res.items(), key=lambda x: x[0])

    def save(self):
        for i, model in enumerate(self.models):
            model.save(os.path.join(self.model_save_dir, self.topology_name[i]), self.epoch_cnt)

        for i, optimizer in enumerate(self.optimizers):
            file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(self.epoch_cnt, i))
            try_mkdir(os.path.split(file_name)[0])
            torch.save(optimizer.state_dict(), file_name)

    def load(self, epoch=None):
        for i, model in enumerate(self.models):
            model.load(os.path.join(self.model_save_dir, self.topology_name[i]), epoch)

        if self.is_train and not self.args.with_end:

            for i, optimizer in enumerate(self.optimizers):
                file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(epoch, i))
                optimizer.load_state_dict(torch.load(file_name))
        self.epoch_cnt = epoch

    def compute_test_result(self):
        mse = torch.nn.MSELoss()
        rec_err = []
        for i in range(self.n_topology):
            gt_pos = self.gt_pos[i]
            rec_pos = self.rec_pos[i]
            rec_err.append(self.criterion_kine(gt_pos, rec_pos, indices=self.models[i].indices_withend))
        cyc_err = []
        p = 0
        for i in range(self.n_topology):
            gt_pos = self.gt_pos[i]
            mean_err = []
            for j in range(self.n_topology):
                if j == i:
                    continue
                cyc_pos = self.cyc_pos[p]
                mean_err.append(mse(gt_pos[..., self.models[i].indices_withend, :],
                                    cyc_pos[..., self.models[i].indices_withend, :]))

                p += 1
            cyc_err.append(torch.mean(torch.Tensor(mean_err)))

        rec_err = torch.Tensor(rec_err)
        cyc_err = torch.Tensor(cyc_err)

        return rec_err, cyc_err, self.fake_pos[0]
