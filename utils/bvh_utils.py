import re
import utils.data_utils as utils
from utils.rotation import *

channelmap = {
    'Xrotation' : 'x',
    'Yrotation' : 'y',
    'Zrotation' : 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x' : 0,
    'y' : 1,
    'z' : 2,
}

class Anim(object):
    """
    A very basic animation object
    """
    def __init__(self, quats, pos, offsets, parents, bones):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones
        self.not_endsite = []
        self.endsite = []
        for i in range(len(bones)):
            if bones[i] != 'End Site':
                self.not_endsite.append(i)
            else:
                self.endsite.append(i)

    @property
    def shape(self): return (self.quats.shape[0], self.quats.shape[1])

    def clip(self, slice):
        self.quats = self.quats[slice]
        self.pos = self.pos[slice]


def read_bvh(filename, start=None, end=None, order=None, downsample_rate=None, start_end=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []

    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)

            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
                # active = parents[active]
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)

            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    if start_end is not None:
        rotations = rotations[start_end[0]: start_end[1]]
        positions = positions[start_end[0]: start_end[1]]

    if downsample_rate is not None:
        Anim_list = []
        for i in range(downsample_rate):
            rotations_tmp = rotations[i::downsample_rate, ...]
            positions_tmp = positions[i::downsample_rate, ...]
            Anim_list.append(Anim(rotations_tmp, positions_tmp, offsets, parents, names))
        return Anim_list
    else:
        return Anim(rotations, positions, offsets, parents, names)


def read_bvh_with_end(filename, start=None, end=None, order=None, downsample_rate=None, start_end=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1

    not_end_index = []
    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            not_end_index.append(active)
            continue

        if "{" in line: continue

        if "}" in line:
            active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            not_end_index.append(active)
            continue

        if "End Site" in line:
            names.append('End Site')
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, not_end_index] = data_block[3:].reshape(len(not_end_index), 3)
                # rotations[fi, np.setdiff1d(np.arange(N), not_end_index)] = \
                #     np.array([[0, 0, 0]]).repeat(N-len(not_end_index), axis=0)
            elif channels == 6:
                data_block = data_block.reshape(len(not_end_index), 6)
                positions[fi, not_end_index] = data_block[:, 0:3]
                rotations[fi, not_end_index] = data_block[:, 3:6]
                # rotations[fi, np.setdiff1d(np.arange(N), not_end_index)] = \
                #     np.array([[0, 0, 0]]).repeat(N - len(not_end_index), axis=0)
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(len(not_end_index) - 1, 9)
                rotations[fi, not_end_index[1:]] = data_block[:, 3:6]
                positions[fi, not_end_index[1:]] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    if start_end is not None:
        rotations = rotations[start_end[0]: start_end[1]]
        positions = positions[start_end[0]: start_end[1]]

    if downsample_rate is not None:
        Anim_list = []
        for i in range(downsample_rate):
            rotations_tmp = rotations[i::downsample_rate, ...]
            positions_tmp = positions[i::downsample_rate, ...]
            Anim_list.append(Anim(rotations_tmp, positions_tmp, offsets, parents, names))
        return Anim_list
    else:
        return Anim(rotations, positions, offsets, parents, names)


def save_bvh(filename, anim, names=None, frametime=1.0 / 24.0, order='zyx',
             positions=False, orients=True, with_end=False,
             end_offset=None, not_end_index=None):
    """
    Saves an Animation to file as BVH

    Parameters
    ----------
    filename: str
        File to be saved to

    anim : Animation
        Animation to save

    names : [str]
        List of joint names

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    frametime : float
        Optional Animation Frame time

    positions : bool
        Optional specfier to save bone
        positions for each frame

    orients : bool
        Multiply joint orients to the rotations
        before saving.

    """

    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]

    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0, 0], anim.offsets[0, 1], anim.offsets[0, 2]))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
                (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        for i in range(anim.shape[1]):
            if anim.parents[i] == 0:
                t, end_offset = save_joint(f, anim, names, t, i, order=order,
                               positions=positions, with_end=with_end, end_offset=end_offset)

        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % anim.shape[0]);
        f.write("Frame Time: %f\n" % frametime);

        # if orients:
        #    rots = np.degrees((-anim.orients[np.newaxis] * anim.rotations).euler(order=order[::-1]))
        # else:
        #    rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        if not_end_index is not None:
            rots = np.degrees(wrap(q2eul, anim.quats[:, not_end_index, :], order[::-1]))
        else:
            rots = np.degrees(wrap(q2eul, anim.quats, order[::-1]))
        poss = anim.pos

        if not_end_index is not None:
            rot_jnum = len(not_end_index)
        else:
            rot_jnum = anim.shape[1]

        for i in range(anim.shape[0]):
            for j in range(rot_jnum):

                if positions or j == 0:
                    f.write("%f %f %f %f %f %f " % (
                        poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                        rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]]))

                else:

                    f.write("%f %f %f " % (
                        rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]]))

            f.write("\n")


def save_joint(f, anim, names, t, i, order='zyx', positions=False, with_end=False, end_offset=None):
    if with_end:
        if names[i] == 'End Site':
            f.write("%s%s\n" % (t, names[i]))
        else:
            f.write("%sJOINT %s\n" % (t, names[i]))
    else:
        f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'

    f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[i, 0], anim.offsets[i, 1], anim.offsets[i, 2]))

    if positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t,
                                                                            channelmap_inv[order[0]],
                                                                            channelmap_inv[order[1]],
                                                                            channelmap_inv[order[2]]))
    else:
        if with_end:
            if names[i] != 'End Site':
                f.write("%sCHANNELS 3 %s %s %s\n" % (t,
                                                     channelmap_inv[order[0]], channelmap_inv[order[1]],
                                                     channelmap_inv[order[2]]))
        else:
            f.write("%sCHANNELS 3 %s %s %s\n" % (t,
                                             channelmap_inv[order[0]], channelmap_inv[order[1]],
                                             channelmap_inv[order[2]]))

    end_site = True

    for j in range(anim.shape[1]):
        if anim.parents[j] == i:
            t, end_offset = save_joint(f, anim, names, t, j, order=order,
                                       positions=positions, with_end=with_end, end_offset=end_offset)
            end_site = False

    if not with_end:
        if end_site:
            f.write("%sEnd Site\n" % t)
            f.write("%s{\n" % t)
            t += '\t'
            if end_offset is None:
                f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
            else:
                f.write("%sOFFSET %f %f %f\n" % (t, end_offset[0, 0], end_offset[0, 1], end_offset[0, 2]))
                end_offset = end_offset[1:, :]
            t = t[:-1]
            f.write("%s}\n" % t)

    t = t[:-1]
    f.write("%s}\n" % t)

    return t, end_offset
