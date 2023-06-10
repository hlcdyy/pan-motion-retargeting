import os, ntpath
import sys
sys.path.append('../../')
from utils.bvh_utils import read_bvh, read_bvh_with_end
from utils import data_utils
from utils.rotation import *


def get_lafan1_set(bvh_dir, actors, choosen_list, window=64, offset=20):
    """
    Extract the motion attributes of lafan1 dataset
    :param bvh_dir: dirctory to the dataset BVH files
    :param actors: actor prefixes to use in set
    :param choosen_list: file names used in extraction
    :param window: width of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps)
    """

    seq_names = []
    X = []
    Q = []
    Joint_velocities = []
    skeleton_names = []
    skeleton_offsets = {}
    skeleton_offsets_withend = {}

    # Extract
    bvh_files = os.listdir(bvh_dir)

    for file in bvh_files:
        if file.endswith('.bvh'):
            base_name = ntpath.basename(file[:-4])
            seq_name, subject = base_name.split('_')

            if subject in actors and base_name in choosen_list:
                print('Processing file {}'.format(file))
                seq_path = os.path.join(bvh_dir, file)
                anim = read_bvh(seq_path)
                anim_withend = read_bvh_with_end(seq_path)
                if not subject in skeleton_offsets.keys():
                    skeleton_offsets[subject] = anim.offsets[1:, ...].ravel()
                    skeleton_offsets_withend[subject] = anim_withend.offsets[1:, ...].ravel()
                # Sliding windows
                i = 0
                while i+window < anim.pos.shape[0] - 1:
                    q, x = data_utils.quat_fk(anim.quats[i: i + window], anim.pos[i: i + window], anim.parents)
                    next_q, next_x = data_utils.quat_fk(anim.quats[i + 1: i + 1 + window], anim.pos[i + 1: i + 1 + window], anim.parents)
                    joint_velocities = next_x - x
                    X.append(anim.pos[i: i+window])
                    Q.append(anim.quats[i: i+window])
                    Joint_velocities.append(joint_velocities)
                    seq_names.append(seq_name)
                    skeleton_names.append(subject)
                    i += offset

    X = np.asarray(X)
    Q = np.asarray(Q)
    Joint_velocities = np.asarray(Joint_velocities)


    # Sequences around XZ = 0
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    X, Q, yrot, forward = data_utils.rotate_at_each_frame(X, Q, anim.parents)

    Joint_velocities = data_utils.extract_local_velocities(Joint_velocities, yrot)
    _, Pos = data_utils.quat_fk(Q, X, anim.parents)

    return X, Q, Pos, Joint_velocities, yrot, \
           skeleton_offsets, skeleton_offsets_withend, \
           skeleton_names, anim.parents, anim_withend.parents, anim_withend.not_endsite


def get_dog_set(bvh_dir, actors, choosen_list, window=64, offset=20):
    """
    Extract the dog dataset like lafan1

    :param bvh_dir: Path to the dataset BVH files
    :param actors: actor prefixes to use in set
    :param choosen_list: choosen sequence list
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps)
    """

    X = []
    Q = []
    Joint_velocities = []
    skeleton_names = []
    skeleton_offsets = {}
    skeleton_offsets_withend = {}

    # Extract
    bvh_files = os.listdir(bvh_dir)

    for file in bvh_files:
        if file.endswith('.bvh'):
            base_name = ntpath.basename(file[:-4])
            split_name = base_name.split('_')
            subject = split_name[0]
            if subject in actors and base_name in choosen_list:
                print('Processing file {}'.format(file))
                seq_path = os.path.join(bvh_dir, file)

                anim_list = read_bvh(seq_path, downsample_rate=2) # fps 60 --> 30 equal to human motion
                anim = anim_list[0]

                anim_withend_list = read_bvh_with_end(seq_path, downsample_rate=2)
                anim_withend = anim_withend_list[0]

                if not subject in skeleton_offsets.keys():
                    skeleton_offsets[subject] = anim.offsets[1:, ...].ravel()
                    skeleton_offsets_withend[subject] = anim_withend.offsets[1:, ...].ravel()

                # Sliding windows
                i = 0
                while i+window < anim.pos.shape[0] - 1:
                    q, x = data_utils.quat_fk(anim.quats[i: i + window], anim.pos[i: i + window], anim.parents)
                    next_q, next_x = data_utils.quat_fk(anim.quats[i + 1: i + 1 + window],
                                                        anim.pos[i + 1: i + 1 + window], anim.parents)
                    joint_velocities = next_x - x
                    X.append(anim.pos[i: i+window])
                    Q.append(anim.quats[i: i+window])
                    Joint_velocities.append(joint_velocities)
                    skeleton_names.append(subject)

                    i += offset

    X = np.asarray(X)
    Q = np.asarray(Q)
    Joint_velocities = np.asarray(Joint_velocities)


    # Sequences around XZ = 0
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # Unify facing
    X, Q, yrot, forward = data_utils.rotate_at_each_dog_frame(X, Q, anim.parents)

    Joint_velocities = data_utils.extract_local_velocities(Joint_velocities, yrot)
    _, Pos = data_utils.quat_fk(Q, X, anim.parents)


    return X, Q, Pos, Joint_velocities, yrot, \
           skeleton_offsets, skeleton_offsets_withend, skeleton_names, \
           anim.parents, anim_withend.parents, anim_withend.not_endsite


def get_lafan1_example(bvh_path):
    """
    Extract the same test set as in the article, given the location of the BVH files.

    :param bvh_path: Path to the dataset BVH files
    :param list: actor prefixes to use in set
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps)
    :return: tuple:
        X: local positions
        Q: local quaternions
        parents: list of parent indices defining the bone hierarchy
    """

    # Extract

    anim = read_bvh(bvh_path)
    anim_withend = read_bvh_with_end(bvh_path)
    frames = anim.pos.shape[0]
    X = anim.pos[:frames]
    Q = anim.quats[:frames]
    q, x = data_utils.quat_fk(anim.quats[:frames], anim.pos[:frames], anim.parents)
    joint_velocities = x[1:, ...] - x[:-1, ...]
    offsets = anim.offsets[1:, ...].ravel()[np.newaxis, ...]
    offsets_withend = anim_withend.offsets[1:, ...].ravel()[np.newaxis, ...]

    xzs = np.mean(X[..., 0, ::2], axis=0, keepdims=True)
    X[..., 0, 0] = X[..., 0, 0] - xzs[..., 0]
    X[..., 0, 2] = X[..., 0, 2] - xzs[..., 1]

    X = X[:-1, ...]
    Q = Q[:-1, ...]

    X, Q, yrot, forward = data_utils.rotate_at_each_frame(X, Q, anim.parents)

    joint_velocities = data_utils.extract_local_velocities(joint_velocities, yrot)
    _, Pos = data_utils.quat_fk(Q, X, anim.parents)

    return X, Q, Pos, joint_velocities, anim.parents, yrot, offsets, offsets_withend


def get_dog_example(bvh_path):
    # Extract

    anim = read_bvh(bvh_path, downsample_rate=2)  # fps 60 --> 30 equal to human motion
    anim = anim[0]
    anim_withend = read_bvh_with_end(bvh_path, downsample_rate=2)  # fps 60 --> 30 equal to human motion
    anim_withend = anim_withend[0]
    offsets = anim.offsets[1:, ...].ravel()[np.newaxis, ...]
    offsets_withend = anim_withend.offsets[1:, ...].ravel()[np.newaxis, ...]


    frames = anim.pos.shape[0]
    X = anim.pos[:frames]
    Q = anim.quats[:frames]

    q, x = data_utils.quat_fk(anim.quats[:frames], anim.pos[:frames], anim.parents)
    joint_velocities = x[1:, ...] - x[:-1, ...]

    # Sequences around XZ = 0
    xzs = np.mean(X[..., 0, ::2], axis=0, keepdims=True)
    X[..., 0, 0] = X[..., 0, 0] - xzs[..., 0]
    X[..., 0, 2] = X[..., 0, 2] - xzs[..., 1]

    X = X[:-1, ...]
    Q = Q[:-1, ...]

    X = X[np.newaxis, ...]
    Q = Q[np.newaxis, ...]
    X, Q, yrot, forward = data_utils.rotate_at_each_dog_frame(X, Q, anim.parents)
    joint_velocities = data_utils.extract_local_velocities(joint_velocities, yrot)
    _, Pos = data_utils.quat_fk(Q, X, anim.parents)


    return X, Q, Pos, joint_velocities, anim.parents, yrot, offsets, offsets_withend


def concatFeature(Q, V, rvel):
    indices = np.where(Q[..., 0] < 0)
    Q[indices] = -Q[indices]
    Q = np.reshape(Q, [Q.shape[0], Q.shape[1], -1])
    V = np.reshape(V, [V.shape[0], V.shape[1], -1])
    RootV = V[..., :3]
    Input = np.concatenate([Q, RootV, rvel], axis=-1)
    return Input


def getStats(Input, joints):
    Input = np.reshape(Input, [-1, Input.shape[-1]])
    mean = Input.mean(axis=0)
    std = Input.std(axis=0)
    root_vel = np.linalg.norm(Input[:, joints * 4: joints * 4 + 3], ord=2, axis=-1)
    min_vel = np.min(root_vel)
    max_vel = np.max(root_vel)
    return mean, std, min_vel, max_vel


if __name__ == '__main__':

    import config as cf
    config = cf.Configuration()

    with open('./data_preprocess/Lafan1_and_dog/dog_train.txt', 'r') as file:
        train_list = file.readlines()
        train_list = [line.strip() for line in train_list]

    dog_dir = './data_preprocess/Lafan1_and_dog/DogSet'
    actors = ['D1']

    X, Q, Pos, V, yrot, \
    skel_offsets, skel_offsets_withend, \
    skel_names, parents, parents_withend, \
    not_endsites = get_dog_set(dog_dir,
                               actors,
                               train_list,
                               window=64,
                               offset=20)

    rvel = wrap(quat2pivots, wrap(qmultipy, wrap(qinv, yrot[:, :-1, ...]), yrot[:, 1:, ...]))
    rvel = np.concatenate((rvel, rvel[:, -1:, ...]), axis=1)
    rvel = np.reshape(rvel, rvel.shape[:2] + (-1,))

    np.savez_compressed('./data_preprocess/Lafan1_and_dog/dogtrain.npz', X=X,
                        Q=Q, Pos=Pos, V=V,
                        parents=parents,
                        yrot=yrot,
                        skel_offsets=skel_offsets,
                        skel_offsets_withend=skel_offsets_withend,
                        skel_names=skel_names
                        )

    with open('./data_preprocess/Lafan1_and_dog/dog_test.txt', 'r') as file:
        test_list = file.readlines()
        test_list = [line.strip() for line in test_list]

    X_test, Q_test, Pos_test, V_test, yrot_test, \
    skel_offsets_test, skel_offsets_withend_test, \
    skel_names_test, parents_test, parents_withend_test, \
    not_endsites_test = get_dog_set(dog_dir,
                               actors,
                               test_list,
                               window=64,
                               offset=20)

    rvel_test = wrap(quat2pivots, wrap(qmultipy, wrap(qinv, yrot_test[:, :-1, ...]), yrot_test[:, 1:, ...]))
    rvel_test = np.concatenate((rvel_test, rvel_test[:, -1:, ...]), axis=1)
    rvel_test = np.reshape(rvel_test, rvel_test.shape[:2] + (-1,))

    np.savez_compressed('./data_preprocess/Lafan1_and_dog/dogtest.npz', X=X_test,
                        Q=Q_test, Pos=Pos_test, V=V_test,
                        parents=parents_test,
                        yrot=yrot_test,
                        skel_offsets=skel_offsets_test,
                        skel_offsets_withend=skel_offsets_withend_test,
                        skel_names=skel_names_test
                        )

    # compute statistics
    Q_all = np.concatenate((Q, Q_test), 0)
    V_all = np.concatenate((V, V_test), 0)
    rvel_all = np.concatenate((rvel, rvel_test), 0)
    Input = concatFeature(Q_all, V_all, rvel_all)
    mean, std, min_vel, max_vel = getStats(Input, config.dog_njoints)
    std = np.where(std == 0, 1, std)
    np.savez_compressed('./data_preprocess/Lafan1_and_dog/dogstats.npz',
                        mean=mean, std=std,
                        min_vel=min_vel, max_vel=max_vel,
                        parents=parents, parents_withend=parents_withend,
                        not_end=not_endsites
                        )

    data_dir = './data_preprocess/Lafan1_and_dog/Lafan1'
    actors = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5']
    with open('./data_preprocess/Lafan1_and_dog/lafan1_train.txt', 'r') as file:
        train_list = file.readlines()
        train_list = [line.strip() for line in train_list]

    X, Q, Pos, V, yrot, \
    skel_offsets, skel_offsets_withend, \
    skel_names, parents, parents_withend, \
    not_endsites = get_lafan1_set(data_dir,
                                  actors,
                                  train_list,
                                  window=64,
                                  offset=20)

    rvel = wrap(quat2pivots, wrap(qmultipy, wrap(qinv, yrot[:, :-1, ...]), yrot[:, 1:, ...]))
    rvel = np.concatenate((rvel, rvel[:, -1:, ...]), axis=1)
    rvel = np.reshape(rvel, rvel.shape[:2] + (-1,))

    np.savez_compressed('./data_preprocess/Lafan1_and_dog/humtrain.npz', X=X,
                        Q=Q, Pos=Pos, V=V,
                        parents=parents,
                        yrot=yrot,
                        skel_offsets=skel_offsets,
                        skel_offsets_withend=skel_offsets_withend,
                        skel_names=skel_names
                        )

    with open('./data_preprocess/Lafan1_and_dog/lafan1_test.txt', 'r') as file:
        test_list = file.readlines()
        test_list = [line.strip() for line in test_list]

    X_test, Q_test, Pos_test, V_test, yrot_test, \
    skel_offsets_test, skel_offsets_withend_test, \
    skel_names_test, parents_test, parents_withend_test, \
    not_endsites_test = get_lafan1_set(data_dir,
                                  actors,
                                  test_list,
                                  window=64,
                                  offset=20)

    rvel_test = wrap(quat2pivots, wrap(qmultipy, wrap(qinv, yrot_test[:, :-1, ...]), yrot_test[:, 1:, ...]))
    rvel_test = np.concatenate((rvel_test, rvel_test[:, -1:, ...]), axis=1)
    rvel_test = np.reshape(rvel_test, rvel_test.shape[:2] + (-1,))

    np.savez_compressed('./data_preprocess/Lafan1_and_dog/humtest.npz', X=X_test,
                        Q=Q_test, Pos=Pos_test, V=V_test,
                        parents=parents_test,
                        yrot=yrot_test,
                        skel_offsets=skel_offsets_test,
                        skel_offsets_withend=skel_offsets_withend_test,
                        skel_names=skel_names_test
                        )

    Q_all = np.concatenate((Q, Q_test), 0)
    V_all = np.concatenate((V, V_test), 0)
    rvel_all = np.concatenate((rvel, rvel_test), 0)
    Input = concatFeature(Q_all, V_all, rvel_all)

    mean, std, min_vel, max_vel = getStats(Input, config.hum_njoints)
    std = np.where(std == 0, 1, std)
    np.savez_compressed('./data_preprocess/Lafan1_and_dog/humstats.npz',
                        mean=mean, std=std,
                        min_vel=min_vel, max_vel=max_vel,
                        parents=parents, parents_withend=parents_withend,
                        not_end=not_endsites
                        )