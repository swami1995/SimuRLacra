import random
import numpy as np
import torch
import ipdb

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.states = []
        self.count = np.zeros(capacity)
        self.position = 0
        self.curv = np.zeros(capacity)
        self.density = np.zeros(capacity)
        self.dist_thres = 0.005

    def push(self, state, action, reward, next_state, done, th_ddot, curv=None):
        for i in range(state.shape[0]):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
                self.states.append(None)
                # self.count.append(0.)
            self.buffer[self.position] = (state[i], action[i], reward[i], next_state[i], done[i], th_ddot[i])
            self.states[self.position] = state[i]
            self.count[self.position] = 0.
            if curv is not None:
                self.curv[self.position] = curv
            self.position = (self.position + 1) % self.capacity


    def push_density(self, state, action, reward, next_state, done, curv=None):
        if len(self.states)>0:
            state1 = torch.stack(self.states)
            dist1 = state.unsqueeze(0) - state1.unsqueeze(1)
            dist1[:, :, 1] /= 8
            dist1[:, :, 0] /= np.pi
            dist1 = dist1[:, :].norm(dim=-1)
            if len(self.states)>=self.capacity:
                dist2 = state1[self.position:self.position + state.shape[0]].unsqueeze(0) - state1.unsqueeze(1)
                dist2[:, :, 1] /= 8
                dist2[:, :, 0] /= np.pi
                dist2 = dist2[:,:].norm(dim=-1)
                dist2_idx = dist2 < self.dist_thres
                self.density[dist2_idx] -= 1
            dist1_idx = (dist1 < self.dist_thres).cpu().numpy()
            dist1_idx[self.position:self.position + state.shape[0]] = False

            # ipdb.set_trace()
            
            self.density[:len(self.states)] += dist1_idx.sum(axis=1)#.cpu().numpy()
            self.density[self.position: self.position + state.shape[0]] = dist1_idx.sum(axis=0)
        dist3 = state.unsqueeze(0) - state.unsqueeze(1)
        dist3[:, :, 1] /= 8
        dist3[:, :, 0] /= np.pi
        dist3 = dist3[:, :].norm(dim=-1)
        dist3_idx = dist3 < self.dist_thres  
        self.density[self.position: self.position + state.shape[0]] += dist3_idx.sum(dim=0).cpu().numpy() - 1
        for i in range(state.shape[0]):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
                self.states.append(None)
                # self.count.append(0.)
            self.buffer[self.position] = (state[i], action[i], reward[i], next_state[i], done[i])
            self.states[self.position] = state[i]
            self.count[self.position] = 0.
            # ipdb.set_trace()
            if curv is not None:
                self.curv[self.position] = curv[i].item()

            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, th_ddot = map(torch.stack, zip(*batch))
        return state, action, reward, next_state, done, th_ddot

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
