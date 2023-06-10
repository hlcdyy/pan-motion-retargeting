import sys
from torch.utils.data.dataloader import DataLoader
from models import create_model_mixamo
from data_preprocess.Mixamo import create_dataset, get_character_names
import parser.parser_mixamo as option_parser
import os
from parser.base import try_mkdir
import time
import torch


torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    args = option_parser.get_args()
    characters = get_character_names(args)

    log_path = os.path.join(args.save_dir, 'logs/')
    try_mkdir(args.save_dir)
    try_mkdir(log_path)

    with open(os.path.join(args.save_dir, 'para.txt'), 'w') as para_file:
        para_file.write(' '.join(sys.argv))

    dataset = create_dataset(args, characters)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = create_model_mixamo(args, characters, dataset)
    if args.use_parallel:
        model.parallel()
    if args.epoch_begin:
        model.load(epoch=args.epoch_begin)

    model.setup()

    start_time = time.time()

    for epoch in range(args.epoch_begin, args.epoch_num):
        for step, motions in enumerate(data_loader):
            model.set_input(motions)  # motions: 0(256, 91, 64)(256) 1(256, 111, 64)(256)
            model.optimize_parameters()

            if args.verbose:
                res = model.verbose()
                print('[{}/{}]\t[{}/{}]\t'.format(epoch, args.epoch_num, step, len(data_loader)), res)

        if epoch % args.save_iter == 0 or epoch == args.epoch_num - 1:
            model.save()

        model.epoch()

    end_tiem = time.time()
    print('training time', end_tiem - start_time)


if __name__ == '__main__':
    main()
