# base parser for biped-quadruped retargeting
from argparse import ArgumentParser

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def add_misc_options(parser):
    group = parser.add_argument_group('Miscellaneous options')
    group.add_argument("--save_dir", help="directory name to save models", default='./run')
    group.add_argument('--with_end', type=boolean_string, default=True, help='whether considering the endsites of the dog')

def add_cuda_options(parser):
    group = parser.add_argument_group('Cuda options')
    group.add_argument('--device', type=str, default='cuda:3')


def adding_cuda(parameters):
    import torch
    if parameters["cuda"] and torch.cuda.is_available():
        parameters["device"] = torch.device("cuda")
    else:
        parameters["device"] = torch.device("cpu")


def add_dataset_options(parser):
    group = parser.add_argument_group('Dataset options')
    group.add_argument("--humstats_path", type=str, default='./data_preprocess/Lafan1_and_dog/humstats.npz')
    group.add_argument("--dogstats_path", type=str, default='./data_preprocess/Lafan1_and_dog/dogstats.npz')
    group.add_argument("--dog_train_path", type=str, default='./data_preprocess/Lafan1_and_dog/dogtrain.npz')
    group.add_argument("--hum_train_path", type=str, default='./data_preprocess/Lafan1_and_dog/humtrain.npz')
    group.add_argument("--hum_test_path", type=str, default='./data_preprocess/Lafan1_and_dog/humtest.npz')
    group.add_argument("--dog_test_path", type=str, default='./data_preprocess/Lafan1_and_dog/dogtest.npz')
    group.add_argument("--time_size", type=int, default=64)


def add_losses_options(parser):
    group = parser.add_argument_group('Losses options')

    group.add_argument("--rec_loss_type", type=str,
                       choices=["mse_rec", "quat_rec", "norm_rec"],
                       default='quat_rec')
    group.add_argument("--root_loss_type", type=str, choices=["mse_root"], default='mse_root')
    group.add_argument("--global_kine_loss_type", type=str,
                       choices=["mse_kine", "l1_kine", "part_kine"], default="part_kine")
    group.add_argument("--cyc_loss_type", type=str, default="mse_cycle_motion")
    group.add_argument("--cyc_latent_loss_type", type=str, default="mse_latent")
    group.add_argument("--retar_vel_loss_type", type=str, default='linear')
    group.add_argument("--dis_loss_type", type=str, choices=["bce_gan", "l2_gan"], default='l2_gan')
    group.add_argument("--retar_vel_matching", type=str, default='mapping', choices=["mapping", 'direct', 'direction'])

    group.add_argument('--lambda_rec', type=float, default=1)
    group.add_argument('--lambda_cycle', type=float, default=1e-3)
    group.add_argument('--lambda_retar_vel', type=float, default=1e3)


def add_model_options(parser):
    group = parser.add_argument_group('Model options')
    group.add_argument("--architecture_name", type=str, default='pan')
    group.add_argument("--fid_net_name", type=str, default='FIDAutoEncoder')

    group.add_argument("--transformer", type=boolean_string, default=True)
    group.add_argument("--transformer_layers", type=int, default=1)
    group.add_argument("--transformer_latents", type=int, default=32)
    group.add_argument("--transformer_ffsize", type=int, default=256)
    group.add_argument("--transformer_heads", type=int, default=1)
    group.add_argument("--transformer_dropout", type=int, default=0)
    group.add_argument("--transformer_srcdim", type=int, default=4)

    group.add_argument("--conv_input", type=int, default=4)
    group.add_argument("--conv_layers", type=int, default=2)
    group.add_argument("--kernel_size", type=int, default=15)
    group.add_argument("--dim_per_part", type=int, default=32)
    group.add_argument("--padding_mode", type=str, default='reflect')

    group.add_argument('--upsampling', type=str, default='linear', help="'stride2' or 'nearest', 'linear'")
    group.add_argument("--skeleton_info", type=str, default="additive")

    group.add_argument("--dis", type=boolean_string, help="use_discriminator", default=True)
    group.add_argument("--diter", type=int, default=3)
    group.add_argument("--dis_mode", type=str,
                        choices=['norm_rotation', 'denorm_rotation', 'denorm_pos', 'latent'], default='denorm_pos')
    group.add_argument("--dis_hidden", type=int, default=256)
    group.add_argument("--dis_layers", type=int, default=3)
    group.add_argument("--dis_kernel_size", type=int, default=15)


def try_mkdir(path):
    import os
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))


class Dict(dict):
    __setattr__ = dict.__setattr__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst