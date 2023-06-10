from models.Intergrated import IntegratedModel_Mixamo
from torch import optim
import torch
from loss_function import GAN_loss, ImagePool, Eval_Criterion, get_ee, Criterion_EE
from models.base_model import BaseModel
from parser.base import try_mkdir
import os


class PAN_model(BaseModel):
    def __init__(self, args, character_names, dataset):
        super(PAN_model, self).__init__(args)
        self.character_names = character_names
        self.dataset = dataset
        self.n_topology = len(character_names)
        self.models = []
        self.D_para = []
        self.G_para = []
        self.args = args

        for i in range(self.n_topology):
            model = IntegratedModel_Mixamo(args, dataset.joint_topologies[i], self.device, character_names[i])
            self.models.append(model)
            self.D_para += model.D_parameters()
            self.G_para += model.G_parameters()

        if self.is_train:
            self.fake_pools = []
            self.optimizerD = optim.Adam(self.D_para, args.learning_rate, betas=(0.9, 0.999))
            self.optimizerG = optim.Adam(self.G_para, args.learning_rate, betas=(0.9, 0.999))
            self.optimizers = [self.optimizerD, self.optimizerG]
            self.criterion_rec = torch.nn.MSELoss()
            self.criterion_gan = GAN_loss(args.gan_mode).to(self.device)
            self.criterion_latent_cycle = torch.nn.L1Loss()
            self.criterion_cycle = torch.nn.MSELoss()
            self.criterion_ee = Criterion_EE(args, torch.nn.MSELoss())

            for i in range(self.n_topology):
                self.fake_pools.append(ImagePool(args.pool_size))
        else:
            import parser.parser_mixamo as option_parser
            self.err_crit = []
            for i in range(self.n_topology):
                self.err_crit.append(Eval_Criterion(dataset.joint_topologies[i]))
            self.id_test = 0
            self.bvh_path = os.path.join(args.save_dir, 'results/bvh')
            option_parser.try_mkdir(self.bvh_path)

            self.writer = []
            for i in range(self.n_topology):
                writer_group = []
                for _, char in enumerate(character_names[i]):
                    from data_preprocess.Mixamo.bvh_writer import BVH_writer
                    from data_preprocess.Mixamo.bvh_parser import BVH_file
                    import parser.parser_mixamo as option_parser
                    file = BVH_file(option_parser.get_std_bvh(dataset=char))
                    writer_group.append(BVH_writer(file.edges, file.names))
                self.writer.append(writer_group)

    def set_input(self, motions):
        self.motions_input = motions

        if not self.is_train:
            self.motion_backup = []
            for i in range(self.n_topology):
                self.motion_backup.append(motions[i][0].clone()) # num_char C T
                self.motions_input[i][0][1:] = self.motions_input[i][0][0]  # only character 0
                self.motions_input[i][1] = [0] * len(self.motions_input[i][1])  #only character 0

    def discriminator_requires_grad_(self, requires_grad):
        for model in self.models:
            for para in model.discriminator.parameters():
                para.requires_grad = requires_grad

    def forward(self):
        self.latents = []
        self.offset_repr = []
        self.pos_ref = []
        self.ee_ref = []
        self.src_offset_repr = []
        self.src_offset_idx = []

        self.res = []
        self.res_denorm = []
        self.res_pos = []
        self.fake_res = []
        self.fake_res_denorm = []
        self.fake_pos = []
        self.fake_ee = []
        self.fake_latent = []

        self.cyc_pos = []

        self.motions = []
        self.raw_offsets = []
        self.motion_denorm = []
        self.rnd_idx = []

        for i in range(self.n_topology):
            self.offset_repr.append(self.models[i].static_encoder(self.dataset.offsets[i]))
            self.raw_offsets.append(self.dataset.offsets[i])

        # reconstruct
        for i in range(self.n_topology):
            motion, offset_idx = self.motions_input[i]
            self.src_offset_idx.append(offset_idx)
            motion = motion.to(self.device)
            self.motions.append(motion)
            motion_denorm = self.dataset.denorm(i, offset_idx, motion)
            self.motion_denorm.append(motion_denorm)
            offsets = self.offset_repr[i][offset_idx]
            self.src_offset_repr.append(offsets)

            latent, res = self.models[i].auto_encoder(motion, offsets)
            res_denorm = self.dataset.denorm(i, offset_idx, res)
            res_pos = self.models[i].fk.forward_from_raw(res_denorm, self.dataset.offsets[i][offset_idx])
            self.res_pos.append(res_pos)
            self.latents.append(latent)
            self.res.append(res)
            self.res_denorm.append(res_denorm)

            pos = self.models[i].fk.forward_from_raw(motion_denorm, self.dataset.offsets[i][offset_idx]).detach()
            ee = get_ee(pos, self.dataset.joint_topologies[i], self.dataset.ee_ids[i],
                        velo=self.args.ee_velo, from_root=self.args.ee_from_root)
            height = self.models[i].height[offset_idx]
            height = height.reshape((height.shape[0], 1, height.shape[1], 1))
            ee /= height
            self.ee_ref.append(ee)
            self.pos_ref.append(pos)


        # retargeting
        for src in range(self.n_topology):
            src_offsets_repr = self.src_offset_repr[src]
            for dst in range(self.n_topology):
                if self.is_train:
                    rnd_idx = torch.randint(len(self.character_names[dst]), (self.latents[src].shape[0],))
                else:
                    rnd_idx = list(range(self.latents[0].shape[0]))
                self.rnd_idx.append(rnd_idx)

                dst_offsets_repr = self.offset_repr[dst][rnd_idx]
                if self.args.use_parallel and self.args.is_train:
                    fake_res = self.models[dst].auto_encoder.module.dec(self.latents[src], dst_offsets_repr)
                else:
                    fake_res = self.models[dst].auto_encoder.dec(self.latents[src], dst_offsets_repr)
                if self.args.use_parallel and self.args.is_train:
                    fake_latent = self.models[dst].auto_encoder.module.enc(fake_res)
                else:
                    fake_latent = self.models[dst].auto_encoder.enc(fake_res)

                fake_res_denorm = self.dataset.denorm(dst, rnd_idx, fake_res)
                fake_pos = self.models[dst].fk.forward_from_raw(fake_res_denorm, self.dataset.offsets[dst][rnd_idx])
                fake_ee = get_ee(fake_pos, self.dataset.joint_topologies[dst], self.dataset.ee_ids[dst],
                                 velo=self.args.ee_velo, from_root=self.args.ee_from_root)
                height = self.models[dst].height[rnd_idx]
                height = height.reshape((height.shape[0], 1, height.shape[1], 1))
                fake_ee = fake_ee / height

                if self.args.use_parallel and self.args.is_train:
                    cyc_res = self.models[src].auto_encoder.module.dec(fake_latent, src_offsets_repr)
                else:
                    cyc_res = self.models[src].auto_encoder.dec(fake_latent, src_offsets_repr)
                cyc_res_denorm = self.dataset.denorm(src, self.src_offset_idx[src], cyc_res)
                cyc_pos = self.models[src].fk.forward_from_raw(cyc_res_denorm, self.dataset.offsets[src][self.src_offset_idx[src]])


                self.fake_latent.append(fake_latent)
                self.fake_pos.append(fake_pos)
                self.fake_ee.append(fake_ee)
                self.fake_res.append(fake_res)
                self.fake_res_denorm.append(fake_res_denorm)
                self.cyc_pos.append(cyc_pos)


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
        self.loss_Ds = []
        self.loss_D = 0
        """
        A->A, A->B, B->A, B->B@[0, 1, 2, 3]
        """
        for i in range(self.n_topology):
            fake = self.fake_pools[i].query(self.fake_pos[2 - i])
            self.loss_Ds.append(self.backward_D_basic(self.models[i].discriminator, self.pos_ref[i].detach(), fake))
            self.loss_D += self.loss_Ds[-1]
            self.loss_recoder.add_scalar('D_loss_{}'.format(i), self.loss_Ds[-1])

    def backward_G(self):
        #rec_loss and gan loss
        self.rec_losses = []
        self.rec_loss = 0
        self.cycle_loss = 0
        self.loss_G = 0
        self.ee_loss = 0
        self.loss_G_total = 0
        for i in range(self.n_topology):
            rec_loss1 = self.criterion_rec(self.motions[i], self.res[i])
            self.loss_recoder.add_scalar('rec_loss_quater_{}'.format(i), rec_loss1)

            height = self.models[i].real_height[self.motions_input[i][1]]
            height = height.reshape(height.shape + (1, 1,))
            input_pos = self.motion_denorm[i][:, -3:, :] / height
            rec_pos = self.res_denorm[i][:, -3:, :] / height
            rec_loss2 = self.criterion_rec(input_pos, rec_pos)
            self.loss_recoder.add_scalar('rec_loss_global_{}'.format(i), rec_loss2)

            pos_ref_global = self.models[i].fk.from_local_to_world(self.pos_ref[i]) / height.reshape(height.shape + (1, ))
            res_pos_global = self.models[i].fk.from_local_to_world(self.res_pos[i]) / height.reshape(height.shape + (1, ))
            rec_loss3 = self.criterion_rec(pos_ref_global, res_pos_global)
            self.loss_recoder.add_scalar('rec_loss_position_{}'.format(i), rec_loss3)

            rec_loss = rec_loss1 + (rec_loss2 + rec_loss3) * 100

            self.rec_losses.append(rec_loss)
            self.rec_loss += rec_loss

        p = 0
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                cycle_latent_loss = self.criterion_latent_cycle(self.latents[src], self.fake_latent[p])

                height = self.models[src].real_height[self.motions_input[src][1]]
                height = height.reshape(height.shape + (1, 1,))
                pos_ref_global = self.models[src].fk.from_local_to_world(self.pos_ref[src]) / height.reshape(
                    height.shape + (1,))
                pos_cyc_global = self.models[src].fk.from_local_to_world(self.cyc_pos[p]) / height.reshape(
                    height.shape + (1,))
                cycle_pos_loss = self.criterion_cycle(pos_ref_global, pos_cyc_global)
                self.loss_recoder.add_scalar('cycle_latent_loss_{}_{}'.format(src, dst), cycle_latent_loss)
                self.loss_recoder.add_scalar('cycle_pos_loss_{}_{}'.format(src, dst), cycle_pos_loss)

                self.cycle_loss += cycle_latent_loss
                self.cycle_loss += cycle_pos_loss

                ee_loss = self.criterion_ee(self.ee_ref[src], self.fake_ee[p])
                self.loss_recoder.add_scalar('ee_loss_{}_{}'.format(src, dst), ee_loss)
                self.ee_loss += ee_loss

                if src != dst:
                    if self.args.gan_mode != 'none':
                        loss_G = self.criterion_gan(self.models[dst].discriminator(self.fake_pos[p]), True)
                    else:
                        loss_G = torch.tensor(0)
                    self.loss_recoder.add_scalar('G_loss_{}_{}'.format(src, dst), loss_G)
                    self.loss_G += loss_G

                p += 1

        self.loss_G_total = self.rec_loss * self.args.lambda_rec + \
                            self.cycle_loss * self.args.lambda_cycle + \
                            self.loss_G * 1 + \
                            self.ee_loss * 50
        self.loss_G_total.backward()

    def optimize_parameters(self):
        self.forward()

        # update Gs
        self.discriminator_requires_grad_(False)
        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        # update Ds
        if self.args.gan_mode != 'none':
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
            model.save(os.path.join(self.model_save_dir, 'topology{}'.format(i)), self.epoch_cnt)

        for i, optimizer in enumerate(self.optimizers):
            file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(self.epoch_cnt, i))
            try_mkdir(os.path.split(file_name)[0])
            torch.save(optimizer.state_dict(), file_name)

    def load(self, epoch=None):
        for i, model in enumerate(self.models):
            model.load(os.path.join(self.model_save_dir, 'topology{}'.format(i)), epoch)
            if not self.is_train:
                model.eval()

        if self.is_train:
            for i, optimizer in enumerate(self.optimizers):
                file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(epoch, i))
                optimizer.load_state_dict(torch.load(file_name))

        self.epoch_cnt = epoch

    def parallel(self):
        for model in self.models:
            model.DataParallel()

    def compute_test_result(self):
        gt_poses = []
        gt_denorm = []
        for src in range(self.n_topology):
            gt = self.motion_backup[src]  # 4 C T
            idx = list(range(gt.shape[0]))
            gt = self.dataset.denorm(src, idx, gt)
            gt_denorm.append(gt)
            gt_pose = self.models[src].fk.forward_from_raw(gt, self.dataset.offsets[src][idx]) # characters, T, J, 3
            gt_poses.append(gt_pose)

            # print(part_weights.shape)   # 2 T parts nodes
            for i in idx:
                new_path = os.path.join(self.bvh_path, self.character_names[src][i])
                try_mkdir(new_path)
                self.writer[src][i].write_raw(gt[i, ...], 'quaternion', os.path.join(new_path, '{}_gt.bvh'.format(self.id_test)))


        p = 0
        for src in range(self.n_topology):
            for dst in range(self.n_topology):
                for i in range(len(self.character_names[dst])):
                    dst_path = os.path.join(self.bvh_path, self.character_names[dst][i])
                    self.writer[dst][i].write_raw(self.fake_res_denorm[p][i, ...], 'quaternion',
                                                  os.path.join(dst_path, '{}_{}.bvh'.format(self.id_test, src)))

                p += 1

        self.id_test += 1
