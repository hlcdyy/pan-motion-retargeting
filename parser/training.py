# training parser for biped-quadruped retargeting
from parser.base import boolean_string, add_misc_options, \
    add_cuda_options, add_dataset_options, \
    add_model_options, add_losses_options, ArgumentParser


def add_training_options(parser):
    group = parser.add_argument_group('Training options')
    group.add_argument('--is_train', type=boolean_string, default=True)
    group.add_argument("--batch_size", type=int, help="size of the batches", default=128)
    group.add_argument("--epoch_begin", type=int, default=0, help="load training epoch !")
    group.add_argument("--epoch_num", type=int, help="number of epochs of training", default=5001)
    group.add_argument("--save_iter", type=int, default=200, help="frequency of saving model/viz per xx epoch")

    group.add_argument("--scheduler", type=str, default='none')
    group.add_argument("--optimizer", type=str, default='Adam')
    group.add_argument('--lr_d', type=float, default=1e-4, help="discriminator learning rate")
    group.add_argument('--lr_g', type=float, default=1e-4, help="generator learning rate")


def get_parser():
    parser = ArgumentParser()
    # misc options
    add_misc_options(parser)

    # cuda options
    add_cuda_options(parser)

    # training options
    add_training_options(parser)

    # dataset options
    add_dataset_options(parser)

    # model options
    add_model_options(parser)

    # loss options
    add_losses_options(parser)

    args = parser.parse_args()
    return args


