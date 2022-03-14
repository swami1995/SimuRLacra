import math
import torch
from numpy import sin, cos, pi
import ipdb
from matplotlib import pyplot as plt 
from operator import itemgetter
from optimizers import newton_al
import numpy as np
import os

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def angle_normalize(x, low=-math.pi, high=math.pi):
    return (((x - low) % (high-low)) + low)

def sample_new_states_ilqr_backward(buffer, counts, curv, density, model, args, device, batch_size, env):
    step_size = torch.tensor([1/5,8/5]).to(device)

    # counts_np = np.hstack(counts)
    counts_np = counts
    probs = np.exp(-counts_np)*np.exp(-density + density.min())#*np.exp(-1/val_errs)
    probs = probs/probs.sum()
    try:
        idxs = np.random.choice(a=len(buffer), size=batch_size*2, replace=False, p=probs)
    except:
        ipdb.set_trace()

    buffer_idxs = list(itemgetter(*idxs)(buffer))
    samples, _, _, _, _ = map(torch.stack, zip(*buffer_idxs))
    rand_noise = torch.randn(samples.shape).to(device)*2 - 1
    if 'pendulum' in args.env_name or 'Pendulum' in args.env_name:
        rand_noise[:,0] *= np.pi/100
        rand_noise[:,1] *= 8/100
    else:
        rand_noise *= np.pi/100
    samples = torch.tensor(samples).to(device) + rand_noise
    samples.requires_grad_(True)
    num_steps = np.random.randint(3) + 1
    # num_steps = 2
    new_samples = find_pre_samples(samples, args, env, nsteps=num_steps)
    if 'pendulum' in args.env_name or 'Pendulum' in args.env_name:
        new_samples[:, 0] = angle_normalize(new_samples[:, 0])
        new_samples[:, 1] = torch.clamp(new_samples[:, 1], -8, 8)
    else:
        new_samples[:, 0] = angle_normalize(new_samples[:, 0], 0, 2*math.pi)
        new_samples[:, 1] = angle_normalize(new_samples[:, 1], -math.pi, math.pi)
        new_samples[:, 2] = torch.clamp(new_samples[:, 2], -4*math.pi, 4*math.pi)
        new_samples[:, 3] = torch.clamp(new_samples[:, 3], -9*math.pi, 9*math.pi)
        
    # self.plot_pairs(samples.detach().cpu().numpy(), new_samples.detach().cpu().numpy())
    # ipdb.set_trace()
    return new_samples.detach().clone(), samples.detach().clone(), torch.zeros_like(new_samples[:, 0]).numpy()

def sample_new_states_ilqr_backward_rrt(buffer, counts, curv, density, model, args, device, batch_size, env, states):
    step_size = torch.tensor([1/5,8/5]).to(device)

    # # counts_np = np.hstack(counts)
    # counts_np = counts
    # probs = np.exp(-counts_np)*np.exp(-density + density.min())#*np.exp(-1/val_errs)
    # probs = probs/probs.sum()
    # try:
    #     idxs = np.random.choice(a=len(buffer), size=batch_size*2, replace=False, p=probs)
    # except:
    #     ipdb.set_trace()

    # buffer_idxs = list(itemgetter(*idxs)(buffer))
    # samples, _, _, _, _ = map(torch.stack, zip(*buffer_idxs))

    rand_states = torch.rand(batch_size*3, env.observation_space.shape[0]).to(env.device)*2 - 1
    if 'pendulum' in args.env_name or 'Pendulum' in args.env_name:
        rand_states[:,0] *= np.pi
        rand_states[:,1] *= 8
    else:
        rand_states[:, 0] *= 1.5*np.pi
        rand_states[:, 0] += np.pi
        rand_states[:, 1] *= np.pi 
        rand_states[:, 2] *= 4*np.pi
        rand_states[:, 3] *= 9*np.pi
    states = torch.stack(states)
    rand_dist = (rand_states.unsqueeze(0) - states.unsqueeze(1)).norm(dim=-1)
    # min_vals, min_dist_args = torch.min(rand_dist, dim=0)
    min_vals, min_dist_args = torch.topk(rand_dist, 20, largest=False, dim=0)
    rand_idxs = torch.randint(20, (3*batch_size,))
    min_vals = min_vals[rand_idxs, torch.arange(3*batch_size)]
    min_dist_args = min_dist_args[rand_idxs, torch.arange(3*batch_size)]
    # ipdb.set_trace()
    samples = states[min_dist_args]
    vals, min_idxs = torch.topk(min_vals, batch_size, largest=False, sorted=False)
    samples = samples[min_idxs]
    rand_states = rand_states[min_idxs]
    env.action_coeffs_vec = torch.clamp((torch.randn((batch_size,1))*0.4)**2, 0,1)*env.action_coeffs

    # ipdb.set_trace()
    args.rand_dirs = rand_states - samples

    rand_noise = torch.randn(samples.shape).to(device)*2 - 1
    if 'pendulum' in args.env_name or 'Pendulum' in args.env_name:
        rand_noise[:,0] *= np.pi/100
        rand_noise[:,1] *= 8/100
    else:
        rand_noise *= np.pi/200
    samples = torch.tensor(samples).to(device) + rand_noise
    samples.requires_grad_(True)
    num_steps = np.random.randint(3) + 1
    # num_steps = 1
    new_samples = find_pre_samples(samples, args, env, nsteps=num_steps)
    if 'pendulum' in args.env_name or 'Pendulum' in args.env_name:
        new_samples[:, 0] = angle_normalize(new_samples[:, 0])
        new_samples[:, 1] = torch.clamp(new_samples[:, 1], -8, 8)
    else:
        new_samples[:, 0] = angle_normalize(new_samples[:, 0], 0, 2*math.pi)
        new_samples[:, 1] = angle_normalize(new_samples[:, 1], -math.pi, math.pi)
        new_samples[:, 2] = torch.clamp(new_samples[:, 2], -4*math.pi, 4*math.pi)
        new_samples[:, 3] = torch.clamp(new_samples[:, 3], -9*math.pi, 9*math.pi)
        
    # self.plot_pairs(samples.detach().cpu().numpy(), new_samples.detach().cpu().numpy())
    # ipdb.set_trace()
    return new_samples.detach().clone(), samples.detach().clone(), torch.zeros_like(new_samples[:, 0]).numpy()

def plot_triplets(samples1, samples2, targs):
    # ipdb.set_trace()
    plt.scatter(samples1[:,0], samples1[:,1], c='r')
    plt.scatter(samples2[:,0], samples2[:,1], c='b')
    plt.scatter(targs[:,0], targs[:,1], c='g')
    for i in range(samples1.shape[0]):
        plt.plot([samples1[i,0], samples2[i,0], targs[i,0]], [samples1[i,1], samples2[i,1], targs[i,1]])
    # plt.quiver(samples1[:,0], samples1[:,1], samples2[:,0]-samples1[:,0], samples2[:,1]-samples1[:,1])
    plt.xlim([-0.3, 0.3])
    plt.ylim([-1, 1])
    plt.savefig('inv_ilqr_rrt_sample_triplets1r0.png')

def plot_pairs(samples1, samples2):
    # ipdb.set_trace()
    plt.scatter(samples1[:,0], samples1[:,1], c='r')
    plt.scatter(samples2[:,0], samples2[:,1], c='b')
    for i in range(samples1.shape[0]):
        plt.plot([samples1[i,0], samples2[i,0]], [samples1[i,1], samples2[i,1]])
    # plt.quiver(samples1[:,0], samples1[:,1], samples2[:,0]-samples1[:,0], samples2[:,1]-samples1[:,1])
    # plt.xlim([-np.pi, np.pi])
    # plt.ylim([-8, 8])
    plt.savefig('inv_ilqr_rrt_sample_pairs1r0.png')

def find_pre_samples(samples, args, env, nsteps=1):
    bsz = samples.shape[0]
    new_samples = torch.stack([samples]*(nsteps+1), dim=1)
    actions_init = torch.zeros((samples.shape[0], nsteps, 1))
    args.rho = 1
    args.rho_ratio = 10
    args.mu_update_tol = 1e-3
    args.rho_max = 1e6
    if args.rand_dirs is None:
        rand_dirs = torch.rand(samples.shape)
        rand_dirs = rand_dirs/(rand_dirs.norm(dim=-1).unsqueeze(-1))
    else:
        rand_dirs = args.rand_dirs
        rand_dirs = rand_dirs/(rand_dirs.norm(dim=-1).unsqueeze(-1))
    prob = torch.rand(rand_dirs.shape[0]).to(rand_dirs)
    mask = (prob < 1).to(prob)
    gradfn_ = lambda xu, mu, rho, sT, xsz, rd: gradfn(xu, mu, rho, sT, xsz, rd, mask, env)
    lagr_func_ = lambda xu, mu, rho, sT, xsz, rd, vec=False: lagr_func(xu, mu, rho, sT, xsz, rd, mask, env, vec)
    result_info = newton_al(gradfn_, lagr_func_, actions_init.reshape(bsz, -1), \
                            new_samples.reshape(bsz, -1),samples, args, rand_dirs)
    next_samples = result_info['result']
    tzero_samples = next_samples[:, :samples.shape[1]]
    return tzero_samples

def gradfn(xu, mu, rho, sT, xsize, rand_dirs, mask, env):
    # xu = torch.autograd.Variable(xu, requires_grad = True)
    diff, lagr = lagr_func(xu, mu, rho, sT, xsize, rand_dirs, mask, env)
    lgrad = torch.autograd.grad(lagr, xu, retain_graph=True, create_graph=True)[0]
    if torch.isnan(lgrad).sum():
        ipdb.set_trace()
    return lgrad, diff, lagr

def lagr_func(xu, mu, rho, sT, xsz, rand_dirs, mask, env, vec=False):
    # We could maximize the dot product with the negative gradient direction w.r.t the value function?
    # Or we could minimize the cumulative sum of rewards but we would probably have to separate the 
    # state rewards and action rewards.
    x, u = xu[:, :xsz], xu[:, xsz:]
    bsz = x.shape[0]
    state_dim = sT.shape[1]
    x1 = x[:, :-state_dim].reshape(-1, state_dim)
    x2 = x[:, state_dim:].reshape(-1, state_dim)
    # u = u.reshape(-1,1)
    nx, r, rs, ra = env(x1, u.reshape(-1,1), return_costs=True, split_costs=True, normalize=False)
    dp = -torch.tanh(((x[:, :state_dim] - sT)*rand_dirs*10).sum(dim=-1))
    rs_sum = rs.reshape(bsz, -1).sum(dim=1)
    rs_sum = rs_sum*(1-mask) + mask*dp
    if torch.isnan(rs).sum() > 0:
        ipdb.set_trace()
    diff = torch.cat([(nx - x2).reshape((bsz, -1)), sT - x[:, -state_dim:]], dim=1)
    action_costs = env.action_coeffs_vec * u**2
    if vec:
        lagr = rs_sum - ra.reshape(bsz, -1).sum(dim=1) + action_costs.sum(dim=1) + (mu * diff).sum(dim=1) + (rho * diff**2).sum(dim=1) + initial_state_cost(x[:, :state_dim], sT)
    else:
        lagr = (rs_sum - ra.reshape(bsz, -1).sum(dim=1)).sum() + action_costs.sum() + (mu * diff).sum() + (rho * diff**2).sum() + initial_state_cost(x[:, :state_dim], sT).sum()
    return diff, lagr

def initial_state_cost(x, sT):
    # need to add density cost and distance cost to the terminal state
    # We could find the k nearest neighbors of sT and use that to find the direction from sT
    # We could maximize the dot product with the negative gradient direction w.r.t the value function?
    # resolving ambuiguity - value gradient dot product
    return torch.zeros_like(x).sum(dim=1)


def samples_from_buffer(buffer, states):
    bsz = states.shape[0]
    idxs = np.random.choice(a=len(buffer), size=bsz, replace=False)
    buffer_idxs = list(itemgetter(*idxs)(buffer))
    samples, _, _, _, _ = map(torch.stack, zip(*buffer_idxs))
    samples = torch.tensor(samples).to(states)
    return samples

def plot_samples(buffer, model, epoch, batch_size, env_name, samples, new_samples):    # sample a bunch of random points
    # 2d plot with value function at each of those points.
    # Compute ground truth value function using MPC cost to go and plot that using the same method. 
    V = []
    states = []
    # model = model.cpu()
    # ipdb.set_trace()
    with torch.no_grad():
    # if True:
        for i in range(len(buffer)//batch_size):
            state1 = [buffer_i[0] for buffer_i in buffer[i*batch_size: (i+1)*batch_size]]
            state = torch.stack(state1)
            # for buffer_i in buffer[i*self.batch_size: (i+1)*self.batch_size]:
                # for j in range(len(buffer_i)):
                # state1[j].append(buffer_i[j])
            # for j in range(len(state1)):
            #     try:
            #         torch.stack(state1[j])
            #     except:
            #         ipdb.set_trace()
            # try:
            #     state, _, _, _, _ = map(torch.stack, buffer[i*self.batch_size: (i+1)*self.batch_size])
            # except:
            #     ipdb.set_trace()
            states.append(state)
            V.append(model(state).squeeze().cpu())
            # if i%20==0:
            #     print(i, ) 
    V = torch.cat(V, dim=0).cpu()*0 + 1
    states = torch.cat(states, dim=0).cpu().detach().requires_grad_(False)
    print("buffer size", states.shape[0])
    # ipdb.set_trace()
    plt.clf()
    plt.scatter(states[:,0], states[:,1], c=V, s=2, cmap='gray')
    plt.scatter(samples[:,0], samples[:,1], c='r', s=2)
    plt.scatter(new_samples[:,0], new_samples[:,1], c='b', s=2)
    for i in range(samples.shape[0]):
        plt.plot([samples[i,0], new_samples[i,0]], [samples[i,1], new_samples[i,1]], linewidth=1)

    if 'pendulum' in env_name:
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-8, 8])
    else:
        plt.xlim([0, 2*np.pi])
        plt.ylim([-np.pi, np.pi])
        
    # path = env_name + 'explore_backward/rrt_lqr_act_cost005_rand200_top20_step13_5200switchonpolicy'
    path = env_name + 'sim2real/sim2real_sac'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + '/val_states_{}.png'.format(epoch))
