from torch.utils.data.dataloader import DataLoader
from models import creat_model
from data_preprocess.Lafan1_and_dog.datasetserial import HumDataset, DogDataset
from parser.training import get_parser
from parser.base import dict_to_object, try_mkdir
import os, sys
from config import Configuration
import torch
from utils.utils import get_body_part


def main():
    args = get_parser()
    parameters_config = {key: val for key, val in vars(Configuration).items() if val is not None}
    parameters_args = {key: val for key, val in vars(args).items() if val is not None}
    parameters_args.update(parameters_config)
    args = dict_to_object(parameters_args)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device[-1]
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    log_path = os.path.join(args.save_dir, 'logs/')
    try_mkdir(args.save_dir)
    try_mkdir(log_path)

    with open(os.path.join(args.save_dir, 'para.txt'), 'w') as para_file:
        para_file.write(' '.join(sys.argv))

    humdataset = HumDataset(args)
    dogdataset = DogDataset(args)

    humloader = DataLoader(humdataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dogloader = DataLoader(dogdataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dogfeeder = iter(dogloader)
    humfeeder = iter(humloader)

    hum_parts = get_body_part(args.correspondence, 'hum_joints')
    dog_parts = get_body_part(args.correspondence, 'dog_joints')

    body_parts = [hum_parts, dog_parts]
    datasets = [humdataset, dogdataset]

    model = creat_model(args, body_parts, datasets, ['human', 'dog'])

    if args.epoch_begin:
        model.load(epoch=args.epoch_begin)

    model.setup()

    epoch = args.epoch_begin
    while epoch < args.epoch_num:
        if epoch % args.save_iter == 0 or epoch == args.epoch_num - 1:
            model.save()

        flag = True
        while flag:
            try:
                input_d, d_yrot, d_offsets, d_offsets_withend = next(dogfeeder)
            except StopIteration:
                dogfeeder = iter(dogloader)
                input_d, d_yrot, d_offsets, d_offsets_withend = next(dogfeeder)

            try:
                input_h, h_yrot, h_offsets, h_offsets_withend = next(humfeeder)
            except StopIteration:
                epoch += 1
                flag = False
                humfeeder = iter(humloader)
                input_h, h_yrot, h_offsets, h_offsets_withend = next(humfeeder)

            vel_dim = 4
            input_h_encoder = (input_h[..., :args.hum_njoints * 4 + vel_dim]).transpose(1, 2)
            input_d_encoder = (input_d[..., :args.dog_njoints * 4 + vel_dim]).transpose(1, 2)

            input_h_encoder = (input_h_encoder, h_offsets, h_offsets_withend)
            input_d_encoder = (input_d_encoder, d_offsets, d_offsets_withend)

            model.set_input([input_h_encoder, input_d_encoder])

            model.optimize_parameters()

        model.epoch()

if __name__ == '__main__':
    main()