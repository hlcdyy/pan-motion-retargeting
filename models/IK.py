from utils.bvh_utils import read_bvh, save_bvh, read_bvh_with_end
from tqdm import tqdm
import torch
import copy
import utils.rotation as rt
import numpy as np
import outer_utils.BVH as BVH
import outer_utils.Animation as Animation
from data_preprocess.Mixamo.bvh_parser import BVH_file
from outer_utils.Quaternions_old import Quaternions
from models.Kinematics import InverseKinematics, InverseKinematics_humdog

L = 6
from scipy import io


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r


def get_character_height(file_name):
    file = BVH_file(file_name)
    return file.get_height()


def get_ee_id_by_names(joint_names, raw_bvh=False):
    if raw_bvh:
        ees = ['RightToe_End', 'LeftToe_End', 'LeftToeBase', 'RightToeBase']
    else:
        ees = ['RightToeBase', 'LeftToeBase', 'LeftFoot', 'RightFoot']

    ee_id = []
    for i, name in enumerate(joint_names):
        if ':' in name:
            joint_names[i] = joint_names[i].split(':')[1]
    for i, ee in enumerate(ees):
        ee_id.append(joint_names.index(ee))
    return ee_id


def get_foot_contact(file_name, ref_height=None, thr=0.003, raw_bvh=False):
    anim, names, _ = BVH.load(file_name)

    ee_ids = get_ee_id_by_names(names, raw_bvh=raw_bvh)

    glb = Animation.positions_global(anim)  # [T, J, 3]

    ee_pos = glb[:, ee_ids, :]
    ee_velo = ee_pos[1:, ...] - ee_pos[:-1, ...]
    if ref_height is not None:
        ee_velo = torch.tensor(ee_velo) / ref_height
    else:
        ee_velo = torch.tensor(ee_velo)
    ee_velo_norm = torch.norm(ee_velo, dim=-1)
    contact = ee_velo_norm < thr
    contact = contact.int()
    padding = torch.zeros_like(contact)
    contact = torch.cat([padding[:1, :], contact], dim=0)
    return contact.numpy()


def remove_foot_sliding(input_file, foot_file, output_file,
                     ref_height, input_raw_bvh=False, foot_raw_bvh=False):

    anim, name, ftime = BVH.load(input_file)
    anim_with_end = read_bvh_with_end(input_file)
    anim_no_end = read_bvh(input_file)

    fid = get_ee_id_by_names(name, input_raw_bvh)
    contact = get_foot_contact(foot_file, ref_height, raw_bvh=foot_raw_bvh)

    glb = Animation.positions_global(anim)  # [T, J, 3]

    T = glb.shape[0]

    for i, fidx in enumerate(fid):  # fidx: index of the foot joint
        fixed = contact[:, i]  # [T]
        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()
            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(L):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(L):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break
            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    # glb is ready

    anim = anim.copy()

    rot = torch.tensor(anim.rotations.qs, dtype=torch.float)
    pos = torch.tensor(anim.positions[:, 0, :], dtype=torch.float)
    offset = torch.tensor(anim.offsets, dtype=torch.float)

    glb = torch.tensor(glb, dtype=torch.float)

    ik_solver = InverseKinematics(rot, pos, offset, anim.parents, glb)

    print('Removing Foot sliding')
    for i in tqdm(range(50)):
        ik_solver.step()

    rotations = ik_solver.rotations.detach()
    norm = torch.norm(rotations, dim=-1, keepdim=True)
    rotations /= norm

    anim.rotations = Quaternions(rotations.numpy())
    anim.positions[:, 0, :] = ik_solver.position.detach().numpy()

    # BVH.save(output_file, anim, name, ftime)
    anim_no_end.quats = anim.rotations.qs
    anim_no_end.pos = anim.positions
    end_offset = anim_with_end.offsets[anim_with_end.endsite, :]

    save_bvh(output_file, anim_no_end, anim_no_end.bones, ftime,
             order='zyx', with_end=False,
             end_offset=end_offset)



def get_foot_contact_by_height2(file_name, end_names, end_site=False):
    if not end_site:
        anim = read_bvh(file_name)
    else:
        anim = read_bvh_with_end(file_name)

    ee_ids = get_ee_id_by_names_humdog(anim.bones, end_names, True)

    _, glb = rt.quat_fk(torch.Tensor(anim.quats), torch.Tensor(anim.pos), anim.parents)
    ee_pos = glb[:, ee_ids, :].numpy()

    contacts = []
    for i in range(0, glb.shape[0], 40):
        end = i+40 if i+40 < glb.shape[0] else glb.shape[0]
        contact = []
        for j in range(len(end_names)):
            min_height = np.min(ee_pos[i: end, j, 1], axis=0)  # len(ee_ids)
            ground_height = min_height + 1.5
            contact.append(ee_pos[i:end, j, 1] < ground_height)
        contact = np.stack(contact, 1)
        contacts.append(contact)

    contacts = np.concatenate(contacts, 0)
    contacts = contacts.astype(np.int)
    if 'Toe' in end_names:
        contacts[:, 2] = contacts[:, 0]
        contacts[:, 3] = contacts[:, 1]

    return contacts


def get_ee_id_by_names_humdog(joint_names, end_names, end_site=False):
    # ees = ['RightHand', 'LeftHand', 'LeftFoot', 'RightFoot']
    ees = end_names
    ee_id = []
    for i, ee in enumerate(ees):
        if end_site:
            ee_id.append(joint_names.index(ee)+1)
        else:
            ee_id.append(joint_names.index(ee))
    return ee_id


def remove_foot_sliding_humdog(input_file, output_file,
                     end_names=['RightHand', 'LeftHand', 'LeftFoot', 'RightFoot'],
                     end_site=False):
    if end_site:
        anim = read_bvh_with_end(input_file)
        end_index = []
        not_end_index = []
        for i in range(len(anim.bones)):
            if anim.bones[i] == 'End Site':
                end_index.append(i)
            else:
                not_end_index.append(i)
        end_offsets = anim.offsets[end_index, :]
    else:
        anim = read_bvh(input_file)

    fid = get_ee_id_by_names_humdog(anim.bones, end_names, end_site)
    contact = get_foot_contact_by_height2(input_file, end_names, end_site)


    _, glb = rt.quat_fk(torch.Tensor(anim.quats), torch.Tensor(anim.pos), anim.parents)
    glb = glb.cpu().numpy()
    T = glb.shape[0]

    for i, fidx in enumerate(fid):  # fidx: index of the foot joint
        fixed = contact[:, i]  # [T]
        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()
            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(L):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(L):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break
            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    # glb is ready
    anim = copy.copy(anim)

    rot = torch.Tensor(anim.quats)
    pos = torch.Tensor(anim.pos[:, 0, :])
    offset = torch.Tensor(anim.offsets)

    glb = torch.Tensor(glb)

    ik_solver = InverseKinematics_humdog(rot, pos, offset, anim.parents, glb)

    print('remove foot sliding using IK...')
    for i in tqdm(range(50)):
        ik_solver.step()

    rotations = ik_solver.rot.detach()
    norm = torch.norm(rotations, dim=-1, keepdim=True)
    rotations /= norm

    anim.quats = rotations.detach().numpy()
    anim.pos[:, 1, :] = ik_solver.pos.detach().numpy()
    if not end_site:
        save_bvh(output_file, anim, frametime=1 / 30, order='zyx', with_end=False,
             names=anim.bones)
    else:
        save_bvh(output_file, anim, frametime=1 / 30, order='zyx', with_end=True,
                 names=anim.bones, end_offset=end_offsets, not_end_index=not_end_index)



def normalize(x):
    return x/torch.norm(x, dim=-1, p=2, keepdim=True)


