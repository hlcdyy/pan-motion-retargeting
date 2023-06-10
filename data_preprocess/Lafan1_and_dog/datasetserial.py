from torch.utils.data import Dataset
from utils.rotation import *


class DogDataset(Dataset):
    def __init__(self, config):
        super(DogDataset, self).__init__()
        self.config = config
        self.njoints = config.dog_njoints
        _data = np.load(config.dog_train_path, allow_pickle=True)
        _stats = np.load(config.dogstats_path, allow_pickle=True)

        self.parents = _stats["parents"]
        self.parents_withend = _stats["parents_withend"]
        self.not_end = _stats["not_end"]
        self.njoints_withend = len(self.parents_withend)

        self.data, self.y_rot, self.skel_offsets, \
        self.skel_offsets_withend, self.skel_names = self._get_in_out(_data)
        self.skel_offsets = self.skel_offsets.item()
        self.skel_offsets_withend = self.skel_offsets_withend.item()
        self.mean, self.std, self.min_vel, self.max_vel = \
            _stats['mean'], _stats['std'], _stats['min_vel'], _stats['max_vel']
        self.data = (self.data - self.mean[np.newaxis, np.newaxis, ...])/self.std[np.newaxis, np.newaxis, ...]

    def _get_in_out(self, _data):
        Q = _data["Q"]
        V = _data["V"]
        yrot = _data["yrot"]
        rvel = wrap(quat2pivots, wrap(qmultipy, wrap(qinv, yrot[:, :-1, ...]), yrot[:, 1:, ...]))
        rvel = np.concatenate((rvel, rvel[:, -1:, ...]), axis=1)
        skel_offsets, skel_offsets_withend, skel_names = \
            _data["skel_offsets"], _data["skel_offsets_withend"], _data["skel_names"]
        indices = np.where(Q[..., 0] < 0)
        Q[indices] = -Q[indices]
        Q = np.reshape(Q, [Q.shape[0], Q.shape[1], -1])
        V = np.reshape(V, [V.shape[0], V.shape[1], -1])
        RootV = V[..., :3]
        Input = np.concatenate([Q, RootV], axis=-1)
        rvel = np.reshape(rvel, rvel.shape[:2] + (-1,))
        Input = np.concatenate([Input, rvel], -1)
        return Input, yrot, skel_offsets, skel_offsets_withend, skel_names

    def __getitem__(self, item):
        skel_name = self.skel_names[item]
        return self.data[item], self.y_rot[item], \
               self.skel_offsets[skel_name], \
               self.skel_offsets_withend[skel_name]

    def __len__(self):
        return len(self.data)

    def denorm(self, x, transpose=False):
        if transpose:
            x = x.transpose(1, 2)
        mean = torch.Tensor(self.mean).to(x.device)
        std = torch.Tensor(self.std).to(x.device)
        b_size, t_size, c_size = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(-1, c_size)
        denorm_x = x * std[:c_size] + mean[:c_size]
        return denorm_x.reshape(b_size, t_size, c_size)


class HumDataset(Dataset):
    def __init__(self, config):
        super(HumDataset, self).__init__()

        hum_path = config.hum_train_path
        self.njoints = config.hum_njoints
        _data = np.load(hum_path, allow_pickle=True)
        _stats = np.load(config.humstats_path, allow_pickle=True)
        self.config = config

        self.parents = _stats["parents"]
        self.parents_withend = _stats["parents_withend"]
        self.not_end = _stats["not_end"]
        self.njoints_withend = len(self.parents_withend)

        self.data, self.y_rot, self.skel_offsets, \
        self.skel_offsets_withend, self.skel_names = self._get_in_out(_data)
        self.skel_offsets = self.skel_offsets.item()
        self.skel_offsets_withend = self.skel_offsets_withend.item()

        self.mean, self.std, self.min_vel, self.max_vel = \
            _stats['mean'], _stats['std'], _stats['min_vel'], _stats['max_vel']

        self.data = (self.data - self.mean[np.newaxis, np.newaxis, ...]) / self.std[np.newaxis, np.newaxis, ...]

    def _get_in_out(self, _data):
        Q = _data["Q"]
        V = _data["V"]
        yrot = _data["yrot"]
        rvel = wrap(quat2pivots, wrap(qmultipy, wrap(qinv, yrot[:, :-1, ...]), yrot[:, 1:, ...]))
        rvel = np.concatenate((rvel, rvel[:, -1:, ...]), axis=1)
        skel_offsets, skel_offsets_withend, skel_names = \
            _data["skel_offsets"], _data["skel_offsets_withend"], _data["skel_names"]
        indices = np.where(Q[..., 0] < 0)
        Q[indices] = -Q[indices]
        Q = np.reshape(Q, [Q.shape[0], Q.shape[1], -1])
        V = np.reshape(V, [V.shape[0], V.shape[1], -1])
        RootV = V[..., :3]
        Input = np.concatenate([Q, RootV], axis=-1)
        rvel = np.reshape(rvel, rvel.shape[:2] + (-1,))
        Input = np.concatenate([Input, rvel], -1)
        return Input, yrot, skel_offsets, skel_offsets_withend, skel_names

    def __getitem__(self, item):
        return self.data[item], self.y_rot[item], \
               self.skel_offsets[self.skel_names[item]], \
               self.skel_offsets_withend[self.skel_names[item]]

    def __len__(self):
        return len(self.data)

    def denorm(self, x, transpose=False):
        if transpose:
            x = x.transpose(1, 2)
        mean = torch.Tensor(self.mean).to(x.device)
        std = torch.Tensor(self.std).to(x.device)
        b_size, t_size, c_size = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(-1, c_size)
        denorm_x = x * std[:c_size] + mean[:c_size]
        return denorm_x.reshape(b_size, t_size, c_size)


class DogDatasetTest(Dataset):
    def __init__(self, config):
        super(DogDatasetTest, self).__init__()

        self.config = config
        self.njoints = config.dog_njoints

        _data = np.load(config.dog_test_path, allow_pickle=True)
        _stats = np.load(config.dogstats_path, allow_pickle=True)

        self.parents = _stats["parents"]
        self.parents_withend = _stats["parents_withend"]
        self.not_end = _stats["not_end"]
        self.njoints_withend = len(self.parents_withend)

        self.data, self.y_rot, self.skel_offsets, \
        self.skel_offsets_withend, self.skel_names = self._get_in_out(_data)
        self.skel_offsets = self.skel_offsets.item()
        self.skel_offsets_withend = self.skel_offsets_withend.item()
        self.mean, self.std, self.min_vel, self.max_vel = \
            _stats['mean'], _stats['std'], _stats['min_vel'], _stats['max_vel']
        self.data = (self.data - self.mean[np.newaxis, np.newaxis, ...]) / self.std[np.newaxis, np.newaxis, ...]

    def _get_in_out(self, _data):
        Q = _data["Q"]
        V = _data["V"]
        yrot = _data["yrot"]
        rvel = wrap(quat2pivots, wrap(qmultipy, wrap(qinv, yrot[:, :-1, ...]), yrot[:, 1:, ...]))
        rvel = np.concatenate((rvel, rvel[:, -1:, ...]), axis=1)
        skel_offsets, skel_offsets_withend, skel_names = \
            _data["skel_offsets"], _data["skel_offsets_withend"], _data["skel_names"]
        indices = np.where(Q[..., 0] < 0)
        Q[indices] = -Q[indices]
        Q = np.reshape(Q, [Q.shape[0], Q.shape[1], -1])
        V = np.reshape(V, [V.shape[0], V.shape[1], -1])
        RootV = V[..., :3]
        Input = np.concatenate([Q, RootV], axis=-1)
        rvel = np.reshape(rvel, rvel.shape[:2] + (-1,))
        Input = np.concatenate([Input, rvel], -1)
        return Input, yrot, skel_offsets, skel_offsets_withend, skel_names

    def __getitem__(self, item):
        return self.data[item], self.y_rot[item], \
               self.skel_offsets[self.skel_names[item]], \
               self.skel_offsets_withend[self.skel_names[item]]

    def __len__(self):
        return len(self.data)

    def denorm(self, x, transpose=False):
        if transpose:
            x = x.transpose(1, 2)
        mean = torch.Tensor(self.mean).to(x.device)
        std = torch.Tensor(self.std).to(x.device)
        b_size, t_size, c_size = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(-1, c_size)
        denorm_x = x * std[:c_size] + mean[:c_size]
        return denorm_x.reshape(b_size, t_size, c_size)


class HumDatasetTest(Dataset):
    def __init__(self, config):
        super(HumDatasetTest, self).__init__()
        _data = np.load(config.hum_test_path, allow_pickle=True)
        _stats = np.load(config.humstats_path, allow_pickle=True)

        self.njoints = config.hum_njoints
        self.config = config

        self.parents = _stats["parents"]
        self.parents_withend = _stats["parents_withend"]
        self.not_end = _stats["not_end"]
        self.njoints_withend = len(self.parents_withend)

        self.data, self.y_rot, self.skel_offsets, \
        self.skel_offsets_withend, self.skel_names = self._get_in_out(_data)
        self.skel_offsets = self.skel_offsets.item()
        self.skel_offsets_withend = self.skel_offsets_withend.item()

        self.mean, self.std, self.min_vel, self.max_vel = \
            _stats['mean'], _stats['std'], _stats['min_vel'], _stats['max_vel']

        self.data = (self.data - self.mean[np.newaxis, np.newaxis, ...]) / self.std[np.newaxis, np.newaxis, ...]

    def _get_in_out(self, _data):
        Q = _data["Q"]
        V = _data["V"]
        yrot = _data["yrot"]
        rvel = wrap(quat2pivots, wrap(qmultipy, wrap(qinv, yrot[:, :-1, ...]), yrot[:, 1:, ...]))
        rvel = np.concatenate((rvel, rvel[:, -1:, ...]), axis=1)
        skel_offsets, skel_offsets_withend, skel_names = \
            _data["skel_offsets"], _data["skel_offsets_withend"], _data["skel_names"]
        indices = np.where(Q[..., 0] < 0)
        Q[indices] = -Q[indices]
        Q = np.reshape(Q, [Q.shape[0], Q.shape[1], -1])
        V = np.reshape(V, [V.shape[0], V.shape[1], -1])
        RootV = V[..., :3]
        Input = np.concatenate([Q, RootV], axis=-1)
        rvel = np.reshape(rvel, rvel.shape[:2] + (-1,))
        Input = np.concatenate([Input, rvel], -1)
        return Input, yrot, skel_offsets, skel_offsets_withend, skel_names

    def __getitem__(self, item):
        return self.data[item], self.y_rot[item], \
               self.skel_offsets[self.skel_names[item]], \
               self.skel_offsets_withend[self.skel_names[item]]

    def __len__(self):
        return len(self.data)

    def denorm(self, x, transpose=False):
        if transpose:
            x = x.transpose(1, 2)
        mean = torch.Tensor(self.mean).to(x.device)
        std = torch.Tensor(self.std).to(x.device)
        b_size, t_size, c_size = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(-1, c_size)
        denorm_x = x * std[:c_size] + mean[:c_size]
        return denorm_x.reshape(b_size, t_size, c_size)