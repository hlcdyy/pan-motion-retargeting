import os
from parser.base import try_mkdir
from utils.rotation import *
from models import creat_model
from data_preprocess.Lafan1_and_dog.datasetserial import HumDataset, DogDataset
from parser.base import dict_to_object
from utils.bvh_utils import save_bvh, Anim, read_bvh, read_bvh_with_end
from data_preprocess.Lafan1_and_dog.extract import get_dog_example
from parser.evaluation import get_parser
from config import Configuration
from utils.utils import get_body_part
from models.IK import remove_foot_sliding_humdog

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# load standard
human_std = './data_preprocess/Lafan1_and_dog/std_bvh/hum_std.bvh'
dog_std = './data_preprocess/Lafan1_and_dog/std_bvh/dog_std.bvh'
std_dog_anim = read_bvh(dog_std)
std_hum_anim = read_bvh(human_std)
standard_pos = std_dog_anim.pos[0:1, ...]
dog_tmp = read_bvh_with_end(dog_std)
hum_tmp = read_bvh_with_end(human_std)
hum_end_sites = []
dog_end_sites = []

for i in range(len(hum_tmp.bones)):
    if hum_tmp.bones[i] == 'End Site':
        hum_end_sites.append(i)

for i in range(len(dog_tmp.bones)):
    if dog_tmp.bones[i] == 'End Site':
        dog_end_sites.append(i)
dog_end_offsets = dog_tmp.offsets[dog_end_sites, :]
hum_end_offsets = hum_tmp.offsets[hum_end_sites, :]

def main():
    bvh_dir = './demo_dir/Dog' # source motion directory
    save_dir = './pretrained_lafan1dog' # save dictory and also the used model dictory.

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    para_path = os.path.join(save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = get_parser(argv_)
    parameters_config = {key: val for key, val in vars(Configuration).items() if val is not None}
    parameters_args = {key: val for key, val in vars(args).items() if val is not None}
    parameters_args.update(parameters_config)
    args = dict_to_object(parameters_args)
    args.device = device
    args.batch_size = 1
    try_mkdir(os.path.join(args.save_dir, 'demo'))

    humdataset = HumDataset(args)
    dogdataset = DogDataset(args)

    hum_parts = get_body_part(args.correspondence, 'hum_joints')
    dog_parts = get_body_part(args.correspondence, 'dog_joints')

    body_parts = [hum_parts, dog_parts]
    datasets = [humdataset, dogdataset]
    model = creat_model(args, body_parts, datasets, ['human', 'dog'])
    model.load(epoch=None)  # specify the epoch number for testing, None is for the latest.
    model.setup()

    ori_name = [name[:-4] for name in os.listdir(bvh_dir) if (name[-4:] == '.bvh' )]
    files = [os.path.join(bvh_dir, name) for name in os.listdir(bvh_dir)
             if (name[-4:] == '.bvh')]
    num = 0

    for file in files:
        print("retargeting the dog motion %s to human skeleton" % str(file))
        X, Q, Pos, V, parents, yrot, offsets, offsets_withend = get_dog_example(file)
        offsets = torch.Tensor(offsets).to(device)
        offsets_withend = torch.Tensor(offsets_withend).to(device)
        rvel = wrap(quat2pivots, wrap(qmultipy, wrap(qinv, yrot[:, :-1, ...]), yrot[:, 1:, ...]))
        rvel = np.concatenate((rvel, rvel[:, -1:, ...]), axis=1)
        rvel = np.reshape(rvel, rvel.shape[:2] + (-1,))

        args.time_size = X.shape[1] - X.shape[1] % 4

        yrot = yrot[:, :args.time_size, ...]
        Q_src = Q.copy()[:, :args.time_size, ...]
        V_src = V.copy()
        Q_src[:, :args.time_size, :1, :] = wrap(qmultipy, yrot, Q[:, :args.time_size, :1, :])
        V_src = wrap(qrot, yrot, V_src[:, :args.time_size])
        for i in range(1, V_src.shape[1]):
            V_src[:, i, ...] = V_src[:, i - 1, ...] + V_src[:, i, ...]
        Pos_src = Pos[:, :args.time_size, ...]
        Pos_src[..., 0, :] = V_src[..., 0, :]

        src_anim = Anim(Q_src.squeeze(), Pos_src.squeeze(),
                       std_dog_anim.offsets, std_dog_anim.parents, std_dog_anim.bones)

        indices = np.where(Q[..., 0] < 0)
        Q[indices] = -Q[indices]
        Q = np.reshape(Q, [Q.shape[0], Q.shape[1], -1])
        V = np.reshape(V, [V.shape[0], V.shape[1], -1])
        RootV = V[..., :3]
        data = np.concatenate([Q, RootV, rvel], axis=-1)
        data = (data - dogdataset.mean[np.newaxis, np.newaxis, ...]) / dogdataset.std[np.newaxis, np.newaxis, ...]


        vel_dim = 4
        input_d_encoder = torch.Tensor(data[..., :args.dog_njoints * 4 + vel_dim]
                                                       ).transpose(1, 2).to(device)

        input_h_encoder = torch.zeros(data.shape[:-1] + (args.hum_njoints * 4 + vel_dim, )
                                      ).transpose(1, 2).to(device) # Placeholder, meaningless

        input_d_encoder = input_d_encoder[..., :args.time_size]
        input_h_encoder = input_h_encoder[..., :args.time_size]

        input_d_encoder = (
            input_d_encoder, offsets, offsets_withend)
        input_h_encoder = (input_h_encoder, torch.zeros(offsets.shape[:-1] + ((args.hum_njoints-1)*3, )
                                                        ).to(device), offsets_withend) # Placeholder, meaningless


        model.set_input([input_h_encoder, input_d_encoder])
        model.forward()

        src, retar = model.motion_denorm[1], model.fake_retar_denorm[1]

        retar_q = qnorm(retar[..., :-vel_dim].reshape(-1, args.hum_njoints, 4))

        retar_vel = retar[..., -4:-1].squeeze()

        retar_q[..., :1, :] = qmultipy(torch.Tensor(yrot).to(device), retar_q[:, :1, :].unsqueeze(0)).squeeze(0)
        retar_vel = qrot(torch.Tensor(yrot).to(device).squeeze(), retar_vel)

        for i in range(1, retar_vel.shape[0]):
            retar_vel[i, ...] = retar_vel[i-1, ...] + retar_vel[i, ...]
        retar_q_np = retar_q.detach().cpu().numpy()
        retar_vel_np = retar_vel.detach().cpu().numpy()[:, np.newaxis]
        pos = standard_pos.repeat(retar_q.shape[0], axis=0)
        pos[:, 0:1, :] = retar_vel_np
        retar_anim = Anim(retar_q_np, pos, std_hum_anim.offsets, std_hum_anim.parents, std_hum_anim.bones)

        if not os.path.exists(os.path.join(args.save_dir, 'demo/dog2hum')):
            os.mkdir(os.path.join(args.save_dir, 'demo/dog2hum'))
        bvh_name = os.path.join(os.path.join(args.save_dir, 'demo/dog2hum'), ori_name[num]+'_retar.bvh')
        save_bvh(bvh_name, retar_anim, frametime=1 / 30, order='zyx', with_end=False,
                 names=retar_anim.bones, end_offset=hum_end_offsets)

        bvh_src = os.path.join(os.path.join(args.save_dir, 'demo/dog2hum'), ori_name[num] + '_source.bvh')
        save_bvh(bvh_src, src_anim, frametime=1 / 30, order='zyx',
                 with_end=False, names=src_anim.bones, end_offset=dog_end_offsets)

        remove_foot_sliding_humdog(bvh_name, bvh_name,
                         end_names=['LeftToe', 'RightToe', 'LeftFoot', 'RightFoot'],
                         end_site=False)

        num += 1

if __name__ == '__main__':
    main()
