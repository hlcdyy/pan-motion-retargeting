import numpy as np
import torch
import torch.nn.functional as F


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def qnorm(q, axis=-1):
    """
    :param q: quaternions to be normalized
    :return:
    """
    q = q/torch.norm(q, dim=axis, keepdim=True)
    return q


def qrot(q, vec):
    assert q.shape[-1] == 4
    assert vec.shape[-1] == 3

    s = q[..., :1]
    u = q[..., 1:]
    u, vec = torch.broadcast_tensors(u, vec)
    uv = torch.cross(u, vec, dim=len(q.shape)-1)
    uuv = torch.cross(u, uv, dim=len(q.shape)-1)
    return (vec+2*(s*uv+uuv)).float()


def qinv(q, in_place=False):
    # we assume q has been normalized
    assert q.shape[-1] == 4
    # q = q/torch.norm(q, dim=-1).unsqueeze(-1)
    if in_place:
        q[...,1:] = -q[...,1:]
        return q
    else:
        w = q[...,:1]
        v = -q[...,1:]
        return torch.cat((w,v),dim=-1)


def quatlog(quat):
    quat = qnorm(quat)
    imgs = quat[..., 1:]
    reals = quat[..., 0]
    lens = torch.sqrt((imgs ** 2).sum(-1))
    lens = torch.atan2(lens, reals) / (lens + 1e-10)
    return imgs * lens.unsqueeze(-1)


def quatexp(ws):
    device = ws.device
    ws = ws.clone()
    ts = (ws ** 2).sum(-1) ** 0.5
    fill = ts.data.new(ts.size()).fill_(0.001)
    ts1 = torch.where(ts == 0, fill, ts)
    ls = torch.sin(ts1) / ts1
    qs = torch.empty(ws.shape[:-1] + (4, )).to(device)
    qs[..., 0] = torch.cos(ts1)
    qs[..., 1] = ws[..., 0] * ls
    qs[..., 2] = ws[..., 1] * ls
    qs[..., 3] = ws[..., 2] * ls

    return qnorm(qs)


def quatbetween(v0, v1):
    assert v0.shape[-1] == 3
    assert v1.shape[-1] == 3
    assert v1.shape[:-1] == v0.shape[:-1]
    shape = tuple(v0.shape[:-1])
    v0 = v0.reshape(-1, 3)
    v1 = v1.reshape(-1, 3)
    a = torch.cross(v0, v1, dim=-1)
    w = (torch.sqrt((v0**2).sum(-1) * (v1 ** 2).sum(-1)) + (v0 * v1).sum(-1)).unsqueeze(-1)

    quaterions = qnorm(torch.cat((w, a), dim=-1)).reshape(shape+(4,))
    return quaterions

def qmultipy(q1, q2):
    """
    in left hand coordinate qmultipy(q1, q2) is equal to q2rotm(q1) * q2rotm(q2)
    :param q1: quaternion q1
    :param q2: quaternion q2
    :return: q1q2
    """
    assert q1.shape[-1] == q2.shape[-1] == 4
    q1, q2 = torch.broadcast_tensors(q1, q2)

    # q1 = q1.double()
    # q2 = q2.double()
    s1 = q1[...,:1]
    s2 = q2[...,:1]
    u1 = q1[...,1:]
    u2 = q2[...,1:]
    w = s1*s2 - torch.sum(u1.mul(u2), dim=-1).unsqueeze(-1)
    v = s1*u2 + s2*u1 + torch.cross(u1, u2, dim=-1)
    return torch.cat((w, v), dim=-1).float()

def quathalf(quat):
    raw_shape = list(quat.shape)
    quat = quat.reshape(-1, 4)
    index = (quat[:, 0] < 0).nonzero(as_tuple=True)[0]
    quat[index, :] = -quat[index, :]
    quat = quat.reshape(tuple(raw_shape[:-1]+[4]))
    return quat


def q2rotm(q):
    flag = False
    if len(q.shape) == 1:
        q = q.unsqueeze_(0)
        flag = True
    raw_shape = list(q.shape)
    q = q.reshape((-1, 4))
    R = torch.zeros([q.shape[0], 9])
    R[:, 0] = 1 - 2*torch.square(q[:, 2])-2*torch.square(q[:, 3])
    R[:, 1] = 2*q[:, 1]*q[:, 2] - 2*q[:, 0]*q[:, 3]
    R[:, 2] = 2*q[:, 1]*q[:, 3] + 2*q[:, 0]*q[:, 2]
    R[:, 3] = 2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3]
    R[:, 4] = 1 - 2*torch.square(q[:, 1])-2*torch.square(q[:, 3])
    R[:, 5] = 2 * q[:, 2] * q[:, 3] - 2 * q[:, 0] * q[:, 1]
    R[:, 6] = 2 * q[:, 1] * q[:, 3] - 2 * q[:, 0] * q[:, 2]
    R[:, 7] = 2 * q[:, 2] * q[:, 3] + 2 * q[:, 0] * q[:, 1]
    R[:, 8] = 1 - 2*torch.square(q[:, 1])-2*torch.square(q[:, 2])
    R = R.reshape(tuple(raw_shape[:-1]+[3, 3]))
    if flag:
        return R.squeeze()
    else:
        return R


def axangle2q(angle):
    """
    :param angle: axis angle with shape(...,3),
    which the norm of vector is rotational theta, and the vector is axis
    :return:quarterion
    """
    flag = False
    if len(angle.shape) == 1:
        flag = True
        angle.unsqueeze_(0)
    raw_shape = list(angle.shape)
    angle = angle.reshape([-1, 3])
    theta = torch.norm(angle, dim=-1).unsqueeze(-1)
    fill = theta.data.new(theta.size()).fill_(0.1)
    theta1 = torch.where(theta == 0, fill, theta)
    axis = angle/theta1
    w = torch.cos(theta/2)
    x = axis[..., :1]*torch.sin(theta/2)
    y = axis[..., 1:2]*torch.sin(theta/2)
    z = axis[..., 2:]*torch.sin(theta/2)
    if flag:
        return torch.cat((w, x, y, z),dim=-1).squeeze()
    else:
        return torch.cat((w, x, y, z),dim=-1).reshape(tuple(raw_shape[:-1]+[4]))

def q2axangle(quat):
    """
    :param quat: quaterion(s) with shape (...,4) (w,x,y,z)
    :return: axis angle(s) (..., 3) (wx,wy,wz),which norm is rotation theta. vector is axis angle.
    """
    assert quat.shape[-1] == 4

    norm_d = torch.norm(quat, dim=-1).unsqueeze(-1)
    quat = quat/norm_d
    w = quat[...,:1]
    xyz = quat[...,1:]
    theta = torch.acos(w)*2
    tmp = torch.sin(theta/2)
    fill = tmp.data.new(tmp.size()).fill_(0.1)
    tmp = torch.where(tmp == 0, fill, tmp)
    xyz = xyz/tmp
    return xyz*theta


def eul2rotm(eulers, order):
    """
    :param eulers: euler angles with shape(..., 3)
    :param order: rotation order
    :return: rotation matrix of coordinates
    """
    flag = False
    if len(eulers.shape) == 1:
        eulers.unsqueeze_(0)
        flag = True
    raw_shape = list(eulers.shape)
    eulers = eulers.reshape((-1, 3))
    ct = torch.cos(eulers)
    st = torch.sin(eulers)
    R = torch.zeros([eulers.shape[0], 9])
    if order.upper() == 'ZYX':
        R[:, 0] = ct[:, 1] * ct[:, 0]
        R[:, 1] = st[:, 2] * st[:, 1] * ct[:, 0] - ct[:, 2] * st[:, 0]
        R[:, 2] = st[:, 1] * ct[:, 2] * ct[:, 0] + st[:, 2] * st[:, 0]
        R[:, 3] = ct[:, 1] * st[:, 0]
        R[:, 4] = st[:, 2] * st[:, 1] * st[:, 0] + ct[:, 2] * ct[:, 0]
        R[:, 5] = ct[:, 2] * st[:, 1] * st[:, 0] - st[:, 2] * ct[:, 0]
        R[:, 6] = -st[:, 1]
        R[:, 7] = ct[:, 1] * st[:, 2]
        R[:, 8] = ct[:, 2] * ct[:, 1]
        R = R.view([-1, 3, 3])
    elif order.upper() == 'ZYZ':
        R[:, 0] = ct[:, 0] * ct[:, 2] * ct[:, 1] - st[:, 0] * st[:, 2]
        R[:, 1] = -ct[:, 0] * ct[:, 1] * st[:, 2] - st[:, 0] * ct[:, 2]
        R[:, 2] = ct[:, 0] * st[:, 1]
        R[:, 3] = st[:, 0] * ct[:, 2] * ct[:, 1] + ct[:, 0] * st[:, 2]
        R[:, 4] = -st[:, 0] * ct[:, 1] * st[:, 2] + ct[:, 0] * ct[:, 2]
        R[:, 5] = st[:, 0] * st[:, 1]
        R[:, 6] = -st[:, 1] * ct[:, 2]
        R[:, 7] = st[:, 1] * st[:, 2]
        R[:, 8] = ct[:, 1]
        R = R.view([-1, 3, 3])
    if flag:
        return R.squeeze()
    else:
        return R.reshape(tuple(raw_shape[:-1]+[3, 3]))

def rotm2eul(rotm, order='zyx'):
    """
    :param rotm: rotation matrix with shape(..., 3, 3)
    :return: euler angles (only support zyx)
    """
    if order.upper() != 'ZYX':
        raise ValueError('Only support zyx rotation order')
    flag = False
    if len(rotm.shape) == 2:
        rotm.unsqueeze_(0)
        flag = True
    raw_shape = list(rotm.shape)
    rotm = rotm.reshape((-1, 3, 3))
    sin_elevation = -rotm[:, 2, 0]
    sy = torch.sqrt(rotm[:, 0, 0] * rotm[:, 0, 0] + rotm[:, 1, 0] * rotm[:, 1, 0])
    sin_azimuth = rotm[:, 1, 0] / sy
    cos_azimuth = rotm[:, 0, 0] / sy

    zero = torch.zeros([rotm.shape[0]])
    one = torch.ones([rotm.shape[0]])
    sin_roll = torch.where(sy > 1e-6, sin_azimuth * rotm[:, 0, 2] - cos_azimuth * rotm[:, 1, 2], zero)
    cos_roll = torch.where(sy > 1e-6, - sin_azimuth * rotm[:, 0, 1] + cos_azimuth * rotm[:, 1, 1], one)

    thetax = torch.atan2(sin_roll, cos_roll).reshape([-1, 1])
    thetay = torch.atan2(sin_elevation, sy).reshape([-1, 1])
    thetaz = torch.atan2(sin_azimuth, cos_azimuth).reshape([-1, 1])
    results = torch.cat([thetaz, thetay, thetax], dim=-1)

    if flag:
        return results.squeeze(0)
    else:
        return results.reshape(tuple(raw_shape[:-2]+[3]))


def angle_axis_to_quat(angle, axis):
    """
    Converts from and angle-axis representation to a quaternion representation

    :param angle: angles tensor
    :param axis: axis tensor
    :return: quaternion tensor
    """
    c = torch.cos(angle / 2.0).unsqueeze(-1)
    s = torch.sin(angle / 2.0).unsqueeze(-1)
    q = torch.cat((c, s * axis), dim=-1)
    # c = np.cos(angle / 2.0)[..., np.newaxis]
    # s = np.sin(angle / 2.0)[..., np.newaxis]
    # q = np.concatenate([c, s * axis], axis=-1)
    return q


def eul2q(e, order='zyx'):
    """
    Converts from an euler representation to a quaternion representation (Coordinates transformation)

    :param e: euler tensor
    :param order: order of euler rotations
    :return: quaternion tensor
    """
    axis = {
        'x': torch.Tensor([1, 0, 0]),
        'y': torch.Tensor([0, 1, 0]),
        'z': torch.Tensor([0, 0, 1]),
        }
    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return qmultipy(q0, qmultipy(q1, q2))


def q2eul(q, order='xyz'):
    q = qnorm(q)
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]
    es = torch.zeros(q.shape[:-1] + (3,))

    if order == 'xyz':
        es[..., 0] = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        es[..., 1] = torch.asin((2 * (q0 * q2 - q3 * q1)).clip(-1, 1))
        es[..., 2] = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        es[..., 0] = torch.atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
        es[..., 1] = torch.atan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
        es[..., 2] = torch.asin((2 * (q1 * q2 + q3 * q0)).clip(-1, 1))
    else:
        raise NotImplementedError('Cannot convert from ordering %s' % order)
    return es

def q2eul_new(q, order='xyz'):
    q = qnorm(q)
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]
    es = torch.zeros(q.shape[:-1] + (3,))

    if order == 'xyz':
        es[..., 2] = torch.atan2(2 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        es[..., 1] = torch.asin((2 * (q1 * q3 + q0 * q2)).clip(-1, 1))
        es[..., 0] = torch.atan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
    elif order == 'yzx':
        es[..., 2] = np.arctan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
        es[..., 1] = np.arcsin((2 * (q1 * q2 + q0 * q3)).clip(-1, 1))
        es[..., 0] = np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
    elif order == 'zxy':
        es[..., 2] = np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        es[..., 1] = np.arcsin((2 * (q0 * q1 + q2 * q3)).clip(-1, 1))
        es[..., 0] = np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
    elif order == 'xzy':
        es[..., 2] = np.arctan2(2 * (q0 * q2 + q1 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        es[..., 1] = np.arcsin((2 * (q0 * q3 - q1 * q2)).clip(-1, 1))
        es[..., 0] = np.arctan2(2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
    elif order == 'yxz':
        es[..., 2] = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
        es[..., 1] = np.arcsin((2 * (q0 * q1 - q2 * q3)).clip(-1, 1))
        es[..., 0] = np.arctan2(2 * (q1 * q3 + q0 * q2), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
    elif order == 'zyx':
        es[..., 2] = np.arctan2(2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        es[..., 1] = np.arcsin((2 * (q0 * q2 - q1 * q3)).clip(-1, 1))
        es[..., 0] = np.arctan2(2 * (q0 * q3 + q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
    else:
        raise NotImplementedError('Cannot convert from ordering %s' % order)
    return es

def rotm2axangle(rotm):
    """
    :param rotm: rotation matrix with shape (...x3x3)
    :return: axis angle(s) (*,3) which norm is rotation theta. vector is axis angle.
    """
    if len(rotm.shape) == 2:
        if (torch.trace(rotm) - 1)/2 >= 1:
            theta = torch.Tensor([0])
        elif (torch.trace(rotm) - 1)/2 <= -1:
            theta = torch.Tensor([np.pi])
        else:
            theta = torch.acos((torch.trace(rotm) - 1)/2)
        if theta == 0:
            theta = torch.Tensor(theta + 0.1)
        ax = torch.stack([rotm[2, 1] - rotm[1, 2], rotm[0,2] - rotm[2,0], rotm[1,0] - rotm[0, 1]], dim=0)
        ax = ax / (2 * torch.sin(theta))
        axangle = ax * theta
    else:
        raw_shape = list(rotm.shape)
        rotm = rotm.reshape([-1, 3, 3])
        theta = torch.from_numpy(np.zeros([rotm.shape[0], 1], dtype=np.float32))
        for batch in range(rotm.shape[0]):
            if (torch.trace(rotm[batch, ...]) - 1) / 2 > 1:
                theta[batch, :] = 0
            elif (torch.trace(rotm[batch, ...]) - 1) / 2 < -1:
                theta[batch, :] = np.pi
            else:
                theta[batch, :] = torch.acos((torch.trace(rotm[batch, ...]) - 1) / 2)
        fill = theta.data.new(theta.size()).fill_(0.1)
        theta = torch.where(theta == 0, fill, theta)
        ax = [rotm[:, 2, 1] - rotm[:, 1, 2], rotm[:, 0, 2] - rotm[:, 2, 0], rotm[:, 1, 0] - rotm[:, 0, 1]]
        ax = torch.stack(ax, dim=1)/(2 * torch.sin(theta))
        axangle = (ax * theta).reshape(tuple(raw_shape[:-2]+[3]))
    return axangle

def axangle2rotm(axangle):
    """
    :param axangle: axis angle(s) (...,3) which norm is rotation theta. vector is axis angle.
    :return: rotation matrixs with shape (...,3,3)
    """
    flag = False
    if len(axangle.shape) == 1:
        axangle.unsqueeze_(0)
        flag = True
    raw_shape = list(axangle.shape)
    axangle = axangle.reshape((-1, 3))
    theta = torch.norm(axangle, dim=-1, keepdim=True)
    ax = axangle/theta
    theta.squeeze_()
    R = torch.zeros([axangle.shape[0], 9])
    R[:, 0] = torch.square(ax[:, 0]) + (1 - torch.square(ax[:, 0])) * torch.cos(theta)
    R[:, 1] = ax[:, 0] * ax[:, 1]*(1-torch.cos(theta)) - ax[:, 2] * torch.sin(theta)
    R[:, 2] = ax[:, 0] * ax[:, 2]*(1-torch.cos(theta)) + ax[:, 1] * torch.sin(theta)
    R[:, 3] = ax[:, 0] * ax[:, 1]*(1-torch.cos(theta)) + ax[:, 2] * torch.sin(theta)
    R[:, 4] = torch.square(ax[:, 1]) + (1 - torch.square(ax[:, 1])) * torch.cos(theta)
    R[:, 5] = ax[:, 1] * ax[:, 2] * (1 - torch.cos(theta)) - ax[:, 0] * torch.sin(theta)
    R[:, 6] = ax[:, 0] * ax[:, 2] * (1 - torch.cos(theta)) - ax[:, 1] * torch.sin(theta)
    R[:, 7] = ax[:, 1] * ax[:, 2] * (1 - torch.cos(theta)) + ax[:, 0] * torch.sin(theta)
    R[:, 8] = torch.square(ax[:, 2]) + (1 - torch.square(ax[:, 2])) * torch.cos(theta)
    if flag:
        R = R.reshape([-1, 3, 3]).squeeze_()
    else:
        R = R.reshape(tuple(raw_shape[:-1]+[3, 3]))
    return R

def direction2pivots(ds, plane='xz'):
    """
    Pivots is an ndarray of angular rotations
    this function mainly for direction to angle
    :param ds: directions
    :param plane: ground plane
    :return:
    """
    ys = ds[..., 'xyz'.index(plane[0])]
    xs = ds[..., 'xyz'.index(plane[1])]
    return torch.atan2(ys, xs).unsqueeze(-1)
    # return np.arctan2(ys, xs)

def quat2pivots(qs, forward='z', plane='xz'):
    ds = torch.zeros(qs.shape[:-1]+(3, ))
    ds[..., 'xyz'.index(forward)] = 1.0
    return direction2pivots(qrot(qs, ds), plane)


def pivots2quat(ps, plane='xz'):
    fa = tuple(map(lambda x: slice(None), ps.shape[:-1]))
    axises = torch.ones(ps.shape[:-1]+(3,)).to(ps.device)
    axises[fa + ("xyz".index(plane[0]), )] = 0.
    axises[fa + ("xyz".index(plane[1]), )] = 0.
    axangle = axises * ps
    return axangle2q(axangle)


def homogeMartrix(R, T):
    """
    :param R: Rotaion matrix with shape (..., 3 * 3)
    :param T: translation with shape (..., 3)
    :return:
    """
    R = F.pad(R, [0, 1, 0, 1])
    T = F.pad(T, [0, 1], mode="constant", value=1)
    R[..., 3] = T
    return R

def orth(R):
    """
    :param R: Rotation matrix with shape (*,3*3)
    :return: Rotation matrix after Schmidt orthogonalization
    """
    if len(R.shape) == 2:
        R = R.unsqueeze_(0)
    results = torch.zeros_like(R)
    for batch in range(R.shape[0]):
        tmp_R = R[batch]
        orth_R = (tmp_R[0, :] / torch.norm(tmp_R[0, :])).reshape([1, -1])
        for i in range(1, tmp_R.shape[0]):
            v = tmp_R[i:i+1, :]
            w = v - torch.matmul(torch.matmul(v, torch.transpose(orth_R, 0, 1)), orth_R)
            orth_R = torch.cat([orth_R, w/torch.norm(w)], dim=0)
        results[batch, ...] = orth_R
    return torch.squeeze(results)

def orth_new(R):
    """
       :param R: Rotation matrix with shape (...,3*3)
       :return: Rotation matrix after Schmidt orthogonalization
       """
    flag = False
    if len(R.shape) == 2:
        R = R.unsqueeze_(0)
        flag = True
    raw_shape = list(R.shape)
    R = R.reshape([-1, 3, 3])
    orth_R = R[:, 0:1, :] / torch.norm(R[:, 0:1, :], dim=2, keepdim=True)
    for i in range(1, R.shape[1]):
        v = R[:, i:i+1, :]
        w = v - torch.matmul(torch.matmul(v, torch.transpose(orth_R, 1, 2)), orth_R)
        orth_R = torch.cat([orth_R, w/torch.norm(w, dim=2, keepdim=True)], dim=1)
    if flag:
        return torch.squeeze(orth_R)
    else:
        return orth_R.reshape(tuple(raw_shape[:-2]+[3, 3]))
    
def rot_rep2rotm(rot_rep):
    """
    :param rot_rep: 6D rotation representation (ref: On the Continuity of Rotation Representations in Neural Networks)
    shape:  xx * 6 or xx * 3 * 2
    :return: rotation matrix
    """
    flag = False
    device = rot_rep.device
    if len(rot_rep.shape) == 1 and rot_rep.shape[-1] == 6:
        rot_rep = rot_rep.unsqueeze(0).reshape(-1, 3, 2)
    elif len(rot_rep.shape) == 2 and rot_rep.shape[-1] == 2 and rot_rep.shape[-2] == 3:
        rot_rep = rot_rep.unsqueeze(0)
    if rot_rep.shape[-1] == 6:
        raw_shape = list(rot_rep.shape)[:-1]
        rot_rep = rot_rep.reshape(-1, 3, 2)
        flag = True
    else:
        raw_shape = list(rot_rep.shape)[:-2]
        rot_rep = rot_rep.reshape(-1, 3, 2)
        flag = True

    orth_R = rot_rep[..., 0:1] / torch.norm(rot_rep[..., :1], dim=-2, keepdim=True)
    for i in range(1, rot_rep.shape[-1]):
        v = rot_rep[..., i:i+1]
        w = v - torch.matmul(orth_R, torch.matmul(torch.transpose(orth_R, 1, 2), v))
        orth_R = torch.cat([orth_R, w/torch.norm(w, dim=-2, keepdim=True)], dim=-1)

    eye = torch.eye(rot_rep.shape[-2]).unsqueeze(0).repeat(orth_R.shape[0], 1, 1).to(device)
    results = torch.zeros(orth_R.shape[0], 3, 3).to(device)
    results[..., :-1] = orth_R
    for i in range(rot_rep.shape[-2]):
        tmp = torch.linalg.det(torch.cat([orth_R, eye[..., i:i+1]], dim=-1))
        results[:, i, -1] = tmp
    if flag:
        results = results.reshape(tuple(raw_shape + [3, 3]))
    else:
        results = results.squeeze()
    return results

def rotm2rot_rep(rotm, flat=False):
    """
    :param rotm: xx * 3 * 3
    :return:
    """
    rep = rotm[..., :, :-1]
    raw_shape = list(rotm.shape[:-2])
    if flat:
        rep = rep.reshape(tuple(raw_shape + [6]))
        return rep
    else:
        return rep


def rotm_fk(lrot, lpos, parents):
    """
    :param lrot: tensor of local rotation matrix with shape (..., Nb of joints, 3, 3)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :].unsqueeze(-1)], [lrot[..., :1, :, :]]
    for i in range(1, len(parents)):
        gp.append(torch.matmul(gr[parents[i]], lpos[..., i:i+1, :].unsqueeze(-1)) + gp[parents[i]])
        gr.append(torch.matmul(gr[parents[i]], lrot[..., i:i+1, :, :]))

    res = torch.cat(gr, dim=-3), torch.cat(gp, dim=-3).squeeze()
    return res

def quat_fk(lrot, lpos, parents):
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    lrot = qnorm(lrot)
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    # parents = parents.int()
    for i in range(1, len(parents)):
        gp.append(qrot(gr[parents[i]], lpos[..., i:i+1, :]) + gp[parents[i]])
        gr.append(qmultipy(gr[parents[i]], lrot[..., i:i+1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)
    return res


def remove_quat_discontinuities(rotations):
    """

    Removing quat discontinuities on the time dimension (removing flips)

    :param rotations: Array of quaternions of shape (T, J, 4)
    :return: The processed array without quaternion inversion.
    """
    rots_inv = -rotations

    for i in range(1, rotations.shape[0]):
        # Compare dot products
        replace_mask = torch.sum(rotations[i - 1: i] * rotations[i: i + 1], dim=-1) < torch.sum(
            rotations[i - 1: i] * rots_inv[i: i + 1], dim=-1)
        replace_mask = replace_mask.unsqueeze(-1)
        rotations[i] = replace_mask * rots_inv[i] + (1.0 - replace_mask.to(int)) * rotations[i]

    return rotations

def local2globalmotion(local_joints, njoints):
    """
    :param local_joints: Batch, Sequence, njoints+1, 3
    :return:
    """
    global_joints = local_joints[..., :njoints, :].clone()
    for i in range(local_joints.shape[1]):
        if i == 0:
            translation = local_joints[:, i, njoints:njoints+1, :].clone()
        else:
            translation = translation + local_joints[:, i, njoints:njoints+1, :].clone()
        global_joints[:, i, ...] = local_joints[:, i, :njoints, :] + translation
    return global_joints

def local2globalvelocity(local_joints, njoints):
    """
    :param local_joints: Batch, Sequence, njoints+1, 3
    :return:
    """
    global_joints = local_joints[..., :njoints, :].clone()
    for i in range(local_joints.shape[1]):
        if i == 0:
            translation = local_joints[:, i, njoints:njoints+1, :].clone()
        else:
            translation = translation + local_joints[:, i, njoints:njoints+1, :].clone()
        global_joints[:, i, ...] = local_joints[:, i, :njoints, :] + translation
    vel = global_joints[:, 1:, ...] - global_joints[:, :-1, ...]
    vel = torch.cat([vel, vel[:, -1:, ...]], dim=1)
    return vel

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

