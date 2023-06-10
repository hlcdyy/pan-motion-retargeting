import numpy as np
from scipy import linalg
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activations(topology_name, args, motion_loader, gan_model, fid_model, datasets, is_san=False):
    print('Calculating Activations...')
    mse = nn.MSELoss()
    errs = []
    activations = []
    topology_names = ['human', 'dog']
    index = topology_names.index(topology_name)
    model = gan_model.models[index]
    fid_net = fid_model.models[index].ae

    if index == 0:
        njoints = args.hum_njoints
    else:
        njoints = args.dog_njoints
    with torch.no_grad():
        if isinstance(motion_loader, DataLoader):
            for idx, batch in enumerate(motion_loader):
                try:
                    input_, yrot, phases = batch
                except:
                    input_, yrot, phases, _ = batch
                batch_input = (input_[..., :njoints * 4 + 3]).transpose(1, 2).float().to(args.device)
                batch_input = datasets[index].denorm(batch_input, transpose=True)

                _, batch_joints = model.fk.forward(batch_input)

                batch_joints_norm = fid_model.normalize(batch_joints.reshape(batch_joints.shape[:2] + (-1, )),
                                                   fid_model.means[index], fid_model.std[index]).transpose(1, 2)
                batch_joints_norm = torch.clone(batch_joints_norm).float().detach_().to(args.device)

                latent, rec = fid_net(batch_joints_norm) # B C 1

                rec_denorm = fid_model.denormalize(rec.transpose(1, 2), fid_model.means[index], fid_model.std[index])
                err = mse(rec_denorm, batch_joints.reshape(batch_joints.shape[:2] + (-1, )))
                errs.append(err)
                activations.append(latent.reshape(latent.shape[0], -1))  # B C T
            activations = torch.cat(activations, dim=0)
            print(torch.mean(torch.stack(errs, 0)))
        else:
            for idx in range(motion_loader.shape[0]):
                if not is_san:
                    input_ = motion_loader[idx]
                    batch_input = torch.clone(input_).float().detach_().to(args.device)
                    batch_input = datasets[index].denorm(batch_input, transpose=True)
                    _, batch_joints = model.fk.forward(batch_input)
                else:
                    batch_joints = motion_loader[idx]
                    batch_joints = torch.from_numpy(batch_joints).to(args.device).float()
                batch_joints = fid_model.normalize(batch_joints.reshape(batch_joints.shape[:2] + (-1,)),
                                                   fid_model.means[index], fid_model.std[index]).transpose(1, 2)
                latent, _ = fid_net(batch_joints)  # B C  B,512
                activations.append(latent.reshape(latent.shape[0], -1))  # B C

            activations = torch.cat(activations, dim=0)

    return activations


def calculate_activation_statistics(activations):
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma

def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])
