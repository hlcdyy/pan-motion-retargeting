import torch
import numpy as np
from utils import rotation as rt


def get_lpos(skel, t_size, njoints, device):
    b_size = skel.shape[0]
    lpos = torch.cat([torch.zeros(b_size, t_size, 3).to(device),
                      skel.unsqueeze(1).repeat(1, t_size, 1)],
                     dim=-1).reshape(b_size, t_size, njoints, -1)  # B T J 3

    return lpos

def get_body_part(correspondence, topology_name):
    part_list = []
    for dic in correspondence:
        part_list.append(dic[topology_name])
    return part_list


def get_part_matrix(part_list, njoints):
    matrix = torch.zeros(len(part_list), njoints)
    for i, part in enumerate(part_list):
        matrix[i, part] = 1
    matrix[:, -1] = 1
    matrix[:, 0] = 1
    return matrix


def get_offset_part_matrix(part_list, num_offsets):
    matrix = torch.zeros(len(part_list), num_offsets+1)
    for i, part in enumerate(part_list):
        matrix[i, part] = 1
    return matrix[:, 1:]


def get_transformer_matrix(part_list, njoints):
    """
    :param part_list: body part list  [[0 ,1 , 2], [1]]   n * 4
    :param njoints: body joints' number plus root velocity
    :return:
    """
    nparts = len(part_list)
    matrix = torch.zeros([nparts + njoints, njoints])

    for i in range(nparts):
        matrix[i, part_list[i]] = 1
        for j in part_list[i]:
            for k in part_list[i]:
                matrix[j + nparts, k] = 1
    matrix[:, 0] = 1
    matrix[:, -1] = 1

    matrix = torch.cat((torch.zeros([njoints + nparts, nparts]), matrix), dim=1)
    for p in range(nparts + njoints):
        matrix[p, p] = 1

    matrix = matrix.float().masked_fill(matrix == 0., float(-1e20)).masked_fill(matrix == 1., float(0.0))
    return matrix


def quat2motion(input, lpos, parents, jnum):
    b_size, t_size = input.shape[0], input.shape[1]
    input_quat = input[..., :jnum * 4].reshape(b_size, t_size, jnum, 4)
    input_vel = input[..., jnum * 4: jnum * 4 + 3].unsqueeze(2)
    _, local_joints = rt.quat_fk(input_quat, lpos, parents)
    return torch.cat([local_joints, input_vel], dim=-2)


def static2motion(local_joints):
    njoints = local_joints.shape[2] - 1
    global_motion = local_joints[..., :njoints, :].clone()
    for i in range(local_joints.shape[1]):
        if i == 0:
            translation = local_joints[:, 0, njoints:, :]
        else:
            translation = local_joints[:, i, njoints:, :] + translation
        global_motion[:, i, ...] = global_motion[:, i, ...] + translation
    return global_motion


def forwardkinematics(input, lpos, parents, jnum):
    local_pos = quat2motion(input, lpos, parents, jnum)
    global_pos = static2motion(local_pos)
    return global_pos, local_pos[..., :-1, :]


class ForwardKinematics:
    def __init__(self, parents, jnum, site_index=None):
        self.parents = parents
        self.jnum = jnum
        self.site_index = site_index

    def forward(self, input, lpos):
        if self.site_index is not None:
            input_new0 = torch.zeros([input.shape[0], input.shape[1], self.jnum, 4]).to(input.device)
            input_new0[..., 0] = 1
            input_new0[..., self.site_index, :] = input[..., :len(self.site_index) * 4].\
                reshape(input.shape[0], input.shape[1], len(self.site_index), 4)
            input_new1 = input[..., len(self.site_index) * 4: len(self.site_index) * 4 + 3]
            input_new = torch.cat((input_new0.reshape(input.shape[0], input.shape[1], -1), input_new1), dim=-1)
            global_pose, local_pose = forwardkinematics(input_new, lpos, self.parents, self.jnum)
        else:
            global_pose, local_pose = forwardkinematics(input, lpos, self.parents, self.jnum)
        return global_pose, local_pose


def findedgechains(edges):
    degree = [0] * 100
    seq_list = []

    for edge in edges:
        degree[edge[0]] += 1
        degree[edge[1]] += 1

    def find_seq(j, seq):
        nonlocal degree, edges, seq_list

        if degree[j] > 2 and j != 0:
            seq_list.append(seq)
            seq = []

        if degree[j] == 1:
            seq_list.append(seq)
            return

        for idx, edge in enumerate(edges):
            if edge[0] == j:
                find_seq(edge[1], seq + [idx])

    find_seq(0, [])
    return seq_list


def findbodychain(edge_seq, edges):
    joint_seq = []
    for seq in edge_seq:
        joint_chain = []
        for i, edge in enumerate(seq):
            joint_chain.append(edges[edge][0])
            if i == len(seq)-1:
                joint_chain.append(edges[edge][1])
        joint_seq.append(joint_chain)
    return joint_seq


def getbodyparts(edges):
    edge_seq = findedgechains(edges)
    joint_seq = findbodychain(edge_seq, edges)
    return joint_seq


def calselfmask(part_list, njoints, edges=None, is_conv=False,
                ):
    part_list = part_list.copy()
    nparts = len(part_list)

    matrix = torch.zeros([njoints + nparts, njoints])
    n = 0

    if edges is not None:
        rotation_map = []
        for i, edge in enumerate(edges):
            rotation_map.append(edge[1])
        rotation_map_reverse = []
        for i in range(1, njoints):
            rotation_map_reverse.append(rotation_map.index(i))

    for part in part_list:
        if part[0] == 0:
            part.pop(0)
        for i in range(len(part)):
            if edges is not None:
                part[i] = rotation_map_reverse[part[i]-1]
            else:
                part[i] -= 1

    for part in part_list:
        matrix[n, part] = 1
        for k in part:
            matrix[k + nparts, part] = 1
        n += 1

    matrix = torch.cat((torch.zeros([njoints+nparts, nparts]), matrix), dim=1)
    for p in range(nparts + njoints):
        matrix[p, p] = 1

    matrix[:, -1] = 1
    if not is_conv:
        matrix = matrix.float().masked_fill(matrix == 0., float(-1e20)).masked_fill(matrix == 1., float(0.0))
    else:
        matrix = matrix[:nparts, nparts:]
    return matrix


def q_mul_q(a, b):
    # sqs, oqs = q_broadcast(a, b)
    sqs, oqs = torch.broadcast_tensors(a, b)
    if sqs.shape[-1] != 4:
        sqs = sqs.reshape(sqs.shape[:-1] + (-1, 4))
        oqs = oqs.reshape(oqs.shape[:-1] + (-1, 4))
    q0 = sqs[..., 0:1]
    q1 = sqs[..., 1:2]
    q2 = sqs[..., 2:3]
    q3 = sqs[..., 3:4]
    r0 = oqs[..., 0:1]
    r1 = oqs[..., 1:2]
    r2 = oqs[..., 2:3]
    r3 = oqs[..., 3:4]

    qs0 = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    qs1 = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    qs2 = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    qs3 = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0

    # return tf.concat([qs0, qs1, qs2, qs3], axis=-1)
    return torch.cat((qs0, qs1, qs2, qs3), -1).reshape(qs0.shape[:-2] + (-1,))

def qnormalize(quat):
    quat = quat.transpose(1, 2)
    b_s, t_s = quat.shape[0], quat.shape[1]
    quat = quat.reshape(b_s, t_s, -1, 4)
    quat = quat/torch.norm(quat).unsqueeze(-1)
    quat = quat.reshape(b_s, t_s, -1).transpose(1, 2)
    return quat

def build_edge_topology(topology, offset):
    # get all edges (pa, child, offset)
    edges = []
    joint_num = len(topology)
    for i in range(1, joint_num):
        edges.append((topology[i], i, offset[i]))
    return edges

def build_joint_topology(edges, origin_names):
    parent = []
    offset = []
    names = []
    edge2joint = []
    joint_from_edge = []  # -1 means virtual joint
    joint_cnt = 0
    out_degree = [0] * (len(edges) + 10)
    for edge in edges:
        out_degree[edge[0]] += 1

    # add root joint
    joint_from_edge.append(-1)
    parent.append(0)
    offset.append(np.array([0, 0, 0]))
    names.append(origin_names[0])
    joint_cnt += 1

    def make_topology(edge_idx, pa):
        nonlocal edges, parent, offset, names, edge2joint, joint_from_edge, joint_cnt
        edge = edges[edge_idx]
        if out_degree[edge[0]] > 1:
            parent.append(pa)
            offset.append(np.array([0, 0, 0]))
            names.append(origin_names[edge[1]] + '_virtual')
            edge2joint.append(-1)
            pa = joint_cnt
            joint_cnt += 1

        parent.append(pa)
        offset.append(edge[2])
        names.append(origin_names[edge[1]])
        edge2joint.append(edge_idx)
        pa = joint_cnt
        joint_cnt += 1

        for idx, e in enumerate(edges):
            if e[0] == edge[1]:
                make_topology(idx, pa)

    for idx, e in enumerate(edges):
        if e[0] == 0:
            make_topology(idx, 0)

    return parent, offset, names, edge2joint
