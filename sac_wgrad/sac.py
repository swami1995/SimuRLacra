import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy, ValueNetwork, QgradNetwork
import ipdb

class SAC(object):
    def __init__(self, num_inputs, action_space, args, env, task_dim, task_vec_optim, task_vec_var):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.env = env
        self.args = args

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, task_dim).to(device=self.device)
        self.vfn = ValueNetwork(num_inputs, args.hidden_size, task_dim).to(device=self.device)
        self.critic_grad = QgradNetwork(num_inputs, action_space.shape[0], args.hidden_size, task_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.val_optim = Adam(self.vfn.parameters(), lr=args.lr)
        self.critic_grad_optim = Adam(self.critic_grad.parameters(), lr=args.lr)
        self.task_vec_optim = task_vec_optim
        self.task_vec_var = task_vec_var

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, task_dim).to(self.device)
        self.critic_grad_target = QgradNetwork(num_inputs, action_space.shape[0], args.hidden_size, task_dim).to(device=self.device)
        hard_update(self.critic_target, self.critic)
        hard_update(self.critic_grad_target, self.critic_grad)
        self.act_coeff = None
        self.state_coeff = None

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            if 'Cartpole' in args.env_name:
                self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, task_dim).to(self.device)
            else:
                self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, task_dim, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            if 'Cartpole' in args.env_name:
                self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, task_dim).to(self.device)
            else:
                self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, task_dim, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, task_vec, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state, task_vec)
        else:
            _, _, action = self.policy.sample(state, task_vec)
        return action.detach().cpu().numpy()[0]

    def select_action_batch(self, state, task_vec, evaluate=False):
        # state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state, task_vec)
        else:
            _, _, action = self.policy.sample(state, task_vec)
        return action.detach()#.cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates, vec, env=None):
        if self.args.grad_explicit:
            return self.update_parameters_explicit_grad(memory, batch_size, updates, vec, env)
        # ipdb.set_trace()
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, th_ddot_batch = memory.sample(batch_size=batch_size)
        if env is None:
            env = self.env

        state_batch = torch.FloatTensor(state_batch).to(self.device).requires_grad_(True)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        th_ddot_batch = torch.FloatTensor(th_ddot_batch).to(self.device)

        pi, log_pi, _ = self.policy.sample(state_batch, vec)
        # action_batch = pi.detach().clone()
        action_batch = action_batch.requires_grad_(True)
        if 'Cartpole' in self.args.env_name:
            next_state_batch, reward_batch, _, _ = env.step_diff_state(state_batch, action_batch, th_ddot_batch)
        else:
            next_state_batch, reward_batch, _, _ = env(state_batch, action_batch, return_costs=True)
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch*0 + 1
        state_dim = state_batch.shape[1]
        # ipdb.set_trace()

        if self.args.grad_vi:
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, vec)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action.detach(), vec)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            # reward_grad = torch.autograd.grad(reward_batch.sum(), state_batch, retain_graph=True)[0]
            # log_pi_grad = torch.autograd.grad(log_pi.sum(), state_batch, retain_graph=True)[0]
            next_q_grad = torch.cat(torch.autograd.grad(next_q_value.sum(), [state_batch, action_batch], retain_graph=True), dim=-1).detach()
            next_q_value = next_q_value.detach()
            # ipdb.set_trace()
        else:
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, vec)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, vec)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # ipdb.set_trace()
        qf1, qf2 = self.critic(state_batch, action_batch, vec)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        if self.args.grad_vi:
            qf1_grad = torch.cat(torch.autograd.grad(qf1.sum(), [state_batch, action_batch],retain_graph=True,create_graph=True), dim=-1)
            qf2_grad = torch.cat(torch.autograd.grad(qf2.sum(), [state_batch, action_batch],retain_graph=True,create_graph=True), dim=-1)
            qf_loss_action = F.mse_loss(qf1_grad[:, state_dim:], next_q_grad[:, state_dim:])
            qf_loss_action += F.mse_loss(qf2_grad[:, state_dim:], next_q_grad[:, state_dim:])
            qf_loss_state = F.mse_loss(qf1_grad[:, :state_dim], next_q_grad[:, :state_dim])            
            qf_loss_state += F.mse_loss(qf2_grad[:, :state_dim], next_q_grad[:, :state_dim])
            if updates % 500==0:
                grad_q = torch.autograd.grad(qf_loss, self.critic.linear3.weight, retain_graph=True)[0].norm()
                grad_action = torch.autograd.grad(qf_loss_action, self.critic.linear3.weight, retain_graph=True)[0].norm()
                grad_state = torch.autograd.grad(qf_loss_state, self.critic.linear3.weight, retain_graph=True)[0].norm()
                print("grad_state, action, Q: ", grad_state, grad_action, grad_q)
                if torch.isnan(grad_state):
                    ipdb.set_trace()
                if updates ==0:
                    self.act_coeff = 1#grad_q/grad_action/2
                    self.state_coeff = 1#grad_q/grad_state/2
                else:
                    self.act_coeff = 0.3*self.act_coeff + 0.7*(grad_q/grad_action/4)
                    self.state_coeff = 0.3*self.act_coeff + 0.7*(grad_q/grad_state)/2

            # qf_loss += F.mse_loss(qf1_grad, next_q_grad)
            # qf_loss += F.mse_loss(qf2_grad, next_q_grad)
            qf_loss += (qf_loss_action)*self.act_coeff + qf_loss_state*self.state_coeff
            # ipdb.set_trace()

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi, vec)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # if self.args.grad_vi:


        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # vf = self.vfn(state_batch, vec)
        # vf_loss = F.mse_loss(vf, (qf1 - self.alpha*log_pi).detach())
        # if self.args.grad_vi:
        #     vf_next = self.vfn(next_state_batch, vec)
        #     next_vf_grad = self.gamma * torch.autograd.grad(vf_next.sum(), state_batch)[0] + \
        #                     reward_grad - self.alpha*log_pi_grad
        #     vf_grad = torch.autograd.grad(vf.sum(), state_batch, retain_graph=True, create_graph=True)[0]
        #     vf_loss += F.mse_loss(next_vf_grad.detach(), vf_grad)

        # self.val_optim.zero_grad()
        # vf_loss.backward()
        # self.val_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def update_parameters_explicit_grad(self, memory, batch_size, updates, vec, env=None):
        # ipdb.set_trace()
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, th_ddot_batch = memory.sample(batch_size=batch_size)
        if env is None:
            env = self.env

        state_batch = torch.FloatTensor(state_batch).to(self.device).requires_grad_(True)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        th_ddot_batch = torch.FloatTensor(th_ddot_batch).to(self.device)

        pi, log_pi, _ = self.policy.sample(state_batch, vec)
        # action_batch = pi.detach().clone()
        action_batch = action_batch.requires_grad_(True)
        if 'Cartpole' in self.args.env_name:
            next_state_batch, reward_batch, _, _ = env.step_diff_state(state_batch, action_batch, th_ddot_batch)
        else:
            next_state_batch, reward_batch, _, _ = env(state_batch, action_batch, return_costs=True)
        reward_batch = reward_batch.unsqueeze(1)
        # mask_batch = mask_batch*0 + 1
        state_dim = state_batch.shape[1]
        # ipdb.set_trace()

        if self.args.grad_vi:
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, vec)
            # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, vec)
            # min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_grad_partial  = self.critic_grad_target(next_state_batch, next_state_action, vec).detach()
            # next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            # ipdb.set_trace()
            next_q_value_for_grad = reward_batch + mask_batch * self.gamma * (next_q_grad_partial * torch.cat([next_state_batch, next_state_action.detach()], dim=-1)).sum(dim=-1).unsqueeze(-1)
            # reward_grad = torch.autograd.grad(reward_batch.sum(), state_batch, retain_graph=True)[0]
            # log_pi_grad = torch.autograd.grad(log_pi.sum(), state_batch, retain_graph=True)[0]
            next_q_grad = torch.clamp(torch.cat(torch.autograd.grad(next_q_value_for_grad.sum(), [state_batch, action_batch], retain_graph=True), dim=-1).detach(), -4,4)
            # test_grads = torch.stack([torch.cat(torch.autograd.grad((torch.cat([next_state_batch, next_state_action.detach()], dim=-1))[:,i].sum(), [state_batch, action_batch], retain_graph=True),dim=-1).detach() for i in range(6)], dim=1)
            # next_q_value = next_q_value.detach()
        else:
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, vec)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, vec)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # ipdb.set_trace()
        # qf1, qf2 = self.critic(state_batch, action_batch, vec)  # Two Q-functions to mitigate positive bias in the policy improvement step
        # qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf_loss = qf1_loss + qf2_loss
        if self.args.grad_vi:
            # qf1_grad = torch.cat(torch.autograd.grad(qf1.sum(), [state_batch, action_batch],retain_graph=True,create_graph=True), dim=-1)
            # qf2_grad = torch.cat(torch.autograd.grad(qf2.sum(), [state_batch, action_batch],retain_graph=True,create_graph=True), dim=-1)
            # qf_loss_action = F.mse_loss(qf1_grad[:, state_dim:], next_q_grad[:, state_dim:])
            # qf_loss_action += F.mse_loss(qf2_grad[:, state_dim:], next_q_grad[:, state_dim:])
            # qf_loss_state = F.mse_loss(qf1_grad[:, :state_dim], next_q_grad[:, :state_dim])            
            # qf_loss_state += F.mse_loss(qf2_grad[:, :state_dim], next_q_grad[:, :state_dim])
            # ipdb.set_trace()
            qf_grad = self.critic_grad(state_batch, action_batch, vec)
            qf_loss_action = F.mse_loss(qf_grad[:, state_dim:], next_q_grad[:, state_dim:])
            qf_loss_state = F.mse_loss(qf_grad[:, :state_dim], next_q_grad[:, :state_dim])
            if updates % 500==0:
                # grad_q = torch.autograd.grad(qf_loss, self.critic.linear3.weight, retain_graph=True)[0].norm()
                grad_action = torch.autograd.grad(qf_loss_action, self.critic_grad.linear3.weight, retain_graph=True)[0].norm()
                grad_state = torch.autograd.grad(qf_loss_state, self.critic_grad.linear3.weight, retain_graph=True)[0].norm()
                print("grad_state, action, Q: ", grad_state, grad_action)
                if torch.isnan(grad_state):
                    ipdb.set_trace()
                if updates ==0:
                    self.act_coeff = 1#grad_q/grad_action/2
                    self.state_coeff = 1#grad_q/grad_state/2
                # else:
                #     self.act_coeff = 0.3*self.act_coeff + 0.7*(grad_state/grad_action)
                    # self.state_coeff = 0.3*self.act_coeff + 0.7*(grad_q/grad_state)/2

            # qf_loss += F.mse_loss(qf1_grad, next_q_grad)
            # qf_loss += F.mse_loss(qf2_grad, next_q_grad)
            qf_loss = (qf_loss_action)*self.act_coeff + qf_loss_state*self.state_coeff
            # ipdb.set_trace()

        self.critic_grad_optim.zero_grad()
        qf_loss.backward()
        self.critic_grad_optim.step()

        # pi, log_pi, _ = self.policy.sample(state_batch)

        # qf1_pi, qf2_pi = self.critic(state_batch, pi, vec)
        qf_grad = self.critic_grad(state_batch, pi, vec).detach()
        # min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - (qf_grad[:,-pi.shape[-1]:]*pi).sum(dim=-1).unsqueeze(-1)).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # if self.args.grad_vi:


        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # vf = self.vfn(state_batch, vec)
        # vf_loss = F.mse_loss(vf, (qf1 - self.alpha*log_pi).detach())
        # if self.args.grad_vi:
        #     vf_next = self.vfn(next_state_batch, vec)
        #     next_vf_grad = self.gamma * torch.autograd.grad(vf_next.sum(), state_batch)[0] + \
        #                     reward_grad - self.alpha*log_pi_grad
        #     vf_grad = torch.autograd.grad(vf.sum(), state_batch, retain_graph=True, create_graph=True)[0]
        #     vf_loss += F.mse_loss(next_vf_grad.detach(), vf_grad)

        # self.val_optim.zero_grad()
        # vf_loss.backward()
        # self.val_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_grad_target, self.critic_grad, self.tau)

        return qf_loss.item(), qf_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def update_parameters_with_real(self, memory, batch_size, updates, vec):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, th_ddot_batch = memory.sample(batch_size=batch_size)


        state_batch = torch.FloatTensor(state_batch).to(self.device).requires_grad_(True)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device).requires_grad_(True)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        th_ddot_batch = torch.FloatTensor(th_ddot_batch).to(self.device)

        vec = torch.stack([self.task_vec_var]*batch_size, dim=0)
        self.task_vec_optim.zero_grad()
        state_dim = state_batch.shape[1]
        # pi, log_pi, _ = self.policy.sample(state_batch, vec)
        # mask_batch = mask_batch*0 + 1
        # ipdb.set_trace()

        if self.args.grad_vi:
            vec_0 = vec*0
            # pi_sim, log_pi_sim, _ = self.policy.sample(state_batch, vec_0)
            action_batch_sim = action_batch# pi_sim.detach()

            mask_batch_sim = mask_batch#*0 + 1 #### NOTE: Might have to change it for negative reward environments
            
            # next_state_batch_sim, reward_batch_sim, _ = self.env(state_batch, action_batch_sim, return_costs=True)
            if 'Cartpole' in self.args.env_name:
                next_state_batch_sim, reward_batch_sim, _, _ = self.env.step_diff_state(state_batch, action_batch_sim, th_ddot_batch)
            else:
                next_state_batch_sim, reward_batch_sim, _ = self.env(state_batch, action_batch_sim, return_costs=True)
            next_state_batch_sim = next_state_batch_sim + (next_state_batch - next_state_batch_sim).detach().clone()
            reward_batch_sim = reward_batch_sim.unsqueeze(1)
            next_state_action_sim, next_state_log_pi_sim, _ = self.policy.sample(next_state_batch_sim, vec_0)
            qf1_next_target_sim, qf2_next_target_sim = self.critic_target(next_state_batch_sim, next_state_action_sim.detach(), vec_0)
            min_qf_next_target_sim = torch.min(qf1_next_target_sim, qf2_next_target_sim) - self.alpha * next_state_log_pi_sim
            next_q_value_sim = reward_batch_sim + mask_batch_sim * self.gamma * (min_qf_next_target_sim)
            # reward_grad_sim = torch.autograd.grad(reward_batch_sim.sum(), [state_batch, action_batch], retain_graph=True)[0]
            # log_pi_grad = torch.autograd.grad(log_pi_sim.sum(), state_batch, retain_graph=True)[0]
            next_q_grad_sim = torch.autograd.grad(next_q_value_sim.sum(), [state_batch,action_batch], retain_graph=True)
            next_q_grad_sim = torch.cat([grad.detach() for grad in next_q_grad_sim], dim=-1)
            next_q_value_sim = next_q_value_sim.detach()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, vec)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, vec)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # ipdb.set_trace()
        qf1, qf2 = self.critic(state_batch, action_batch, vec)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        if self.args.grad_vi:
            qf1_sim, qf2_sim = self.critic(state_batch, action_batch_sim, vec_0)
            qf1_grad_sim = torch.cat(torch.autograd.grad(qf1_sim.sum(), [state_batch, action_batch],retain_graph=True,create_graph=True), dim=-1)
            qf2_grad_sim = torch.cat(torch.autograd.grad(qf2_sim.sum(), [state_batch, action_batch],retain_graph=True,create_graph=True), dim=-1)
            qf_loss_action = F.mse_loss(qf1_grad_sim[:, state_dim:], next_q_grad_sim[:, state_dim:])
            qf_loss_action += F.mse_loss(qf2_grad_sim[:, state_dim:], next_q_grad_sim[:, state_dim:])
            qf_loss_state = F.mse_loss(qf1_grad_sim[:, :state_dim], next_q_grad_sim[:, :state_dim])            
            qf_loss_state += F.mse_loss(qf2_grad_sim[:, :state_dim], next_q_grad_sim[:, :state_dim])
            if updates % 500==0:
                grad_q = torch.autograd.grad(qf_loss, self.critic.linear3.weight, retain_graph=True)[0].norm()
                grad_action = torch.autograd.grad(qf_loss_action, self.critic.linear3.weight, retain_graph=True)[0].norm()
                grad_state = torch.autograd.grad(qf_loss_state, self.critic.linear3.weight, retain_graph=True)[0].norm()
                print("grad_state, action, Q: ", grad_state, grad_action, grad_q)
                if updates ==0 or self.act_coeff is None:
                    self.act_coeff = grad_q/grad_action/4
                    self.state_coeff = grad_q/grad_state/2
                else:
                    self.act_coeff = 0.3*self.act_coeff + 0.7*(grad_q/grad_action/4)
                    self.state_coeff = 0.3*self.act_coeff + 0.7*(grad_q/grad_state)/2

            # qf_loss += F.mse_loss(qf1_grad, next_q_grad)
            # qf_loss += F.mse_loss(qf2_grad, next_q_grad)
            qf_loss += (qf_loss_action)*self.act_coeff + qf_loss_state*self.state_coeff
            # ipdb.set_trace()

            # qf_loss += F.mse_loss(qf1_grad_sim, next_q_grad_sim)#*200
            # qf_loss += F.mse_loss(qf2_grad_sim, next_q_grad_sim)#*200

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        # self.task_vec_optim.step()

        vec = torch.stack([self.task_vec_var]*batch_size, dim=0)
        self.task_vec_optim.zero_grad()
        pi, log_pi, _ = self.policy.sample(state_batch, vec)

        qf1_pi, qf2_pi = self.critic(state_batch, pi, vec.detach())
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # if self.args.grad_vi:


        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        # self.task_vec_optim.step()

        # vf = self.vfn(state_batch, vec.detach())
        # vf_loss = F.mse_loss(vf, (qf1 - self.alpha*log_pi).detach())
        # if self.args.grad_vi:
        #     vf_sim = self.vfn(state_batch, vec_0.detach())
        #     vf_next_sim = self.vfn(next_state_batch_sim, vec_0.detach())
        #     next_vf_grad = self.gamma * torch.autograd.grad(vf_next_sim.sum(), state_batch)[0] + \
        #                     reward_grad_sim - self.alpha*log_pi_grad
        #     vf_grad = torch.autograd.grad(vf_sim.sum(), state_batch, retain_graph=True, create_graph=True)[0]
        #     vf_loss += F.mse_loss(next_vf_grad.detach(), vf_grad)

        # self.val_optim.zero_grad()
        # vf_loss.backward()
        # self.val_optim.step()


        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    'vfn_state_dict': self.vfn.state_dict(),
                    'vfn_optimizer_state_dict': self.val_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.vfn.load_state_dict(checkpoint['vfn_state_dict'])
            self.val_optim.load_state_dict(checkpoint['vfn_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

