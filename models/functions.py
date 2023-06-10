import loss_function as lf
import torch.nn as nn
import torch.optim as optim

def get_gan_loss(loss_type):
    if loss_type == 'bce_gan':
        return lf.dis_loss
    elif loss_type == 'l2_gan':
        return lf.dis_loss_l2

def get_rec_loss(loss_type):
    if loss_type == 'mse_rec':
        return nn.MSELoss()
    elif loss_type == 'quat_rec' or loss_type == 'norm_rec':
        return lf.caloutputloss

def get_root_loss(loss_type):
    if loss_type == 'mse_root':
        return nn.MSELoss()

def get_kine_loss(loss_type):
    if loss_type == 'mse_kine':
        return nn.MSELoss()
    elif loss_type == 'part_kine':
        return lf.calposloss

def get_cycle_loss(loss_type):
    if loss_type == 'mse_cycle_motion':
        return lf.cycle_motions

def get_cycle_latent_loss(loss_type):
    if loss_type == 'mse_latent':
        return lf.cycle_latents

def get_retar_root_v_loss(loss_type):
    if loss_type == 'linear':
        return nn.MSELoss()


def get_discriminator_input(gan_model, dis_mode, index, real):
    if dis_mode == 'norm_rotation':
        if real:
            return gan_model.motion[index]
        else:
            return gan_model.fake_retar[index].transpose(1, 2)
    elif dis_mode == 'denorm_rotation':
        if real:
            return gan_model.motion_denorm[index].transpose(1, 2)
        else:
            return gan_model.fake_retar_denorm[index].transpose(1, 2)
    elif dis_mode == 'denorm_pos':
        if real:
            return gan_model.gt_pos[index].reshape(gan_model.gt_pos[index].shape[:-2] + (-1, )).transpose(1, 2)
        else:
            return gan_model.fake_pos[index].reshape(gan_model.fake_pos[index].shape[:-2] + (-1, )).transpose(1, 2)
    elif dis_mode == 'latent':
        if real:
            return gan_model.latents[index]
        else:
            return gan_model.retar_latents[index]
    else:
        raise Exception("Discriminator input not defined !")


def get_recloss_input(gan_model, rec_loss_mode, index):
    if rec_loss_mode == 'norm_rec':
        input_0 = gan_model.motion[index].transpose(1, 2)
        input_1 = gan_model.rec[index]
    elif rec_loss_mode == 'quat_rec':
        input_0 = gan_model.motion_denorm[index]
        input_1 = gan_model.rec_denorm[index]
    return input_0, input_1


def get_retar_latents(gan_model, src):
    input = gan_model.latents[src]
    out_latent = input
    return out_latent


def get_cyc_latents(gan_model, fake_latent):

    # Construct shared latent space without any transfer
    out_latent = fake_latent
    return out_latent


def get_optimizer(optimizer_name, para, lr, **kwargs):
    if optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(para, lr=lr)

    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(para, lr=lr, **kwargs)

    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(para, lr=lr, **kwargs)

    else:
        raise Exception("Optimizer not find!")

    return optimizer


