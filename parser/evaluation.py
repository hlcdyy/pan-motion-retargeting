# evaluation parser for biped-quadruped retargeting
from parser.base import boolean_string, add_misc_options, \
    add_cuda_options, ArgumentParser, add_dataset_options, add_model_options, add_losses_options

def add_evaluation_options(parser):
    group = parser.add_argument_group('Evaluation options')
    group.add_argument('--is_train', type=boolean_string, default=False)
    group.add_argument('--batch_size', type=int, help="size of the batches", default=24)
    group.add_argument('--verbose', type=boolean_string, default=True)
    group.add_argument('--epoch_begin', type=int, default=0)
    group.add_argument("--save_iter", type=int, default=200, help="frequency of saving model/viz per xx epoch")
    group.add_argument("--epoch_num", type=int, default=4001)



def get_parser(argv_=None):
    parser = ArgumentParser()
    # misc options
    add_misc_options(parser)

    # cuda options
    add_cuda_options(parser)

    # training options
    add_evaluation_options(parser)

    # dataset options
    add_dataset_options(parser)

    # model options
    add_model_options(parser)

    # loss options
    add_losses_options(parser)

    args = parser.parse_args(argv_)
    return args

