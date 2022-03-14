import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
from envs import PendulumDynamics, AcrobotEnv, SGAcroEnvWrapper
from utils import *
from torch.optim import Adam
import ipdb
# from pyrado.algorithms.step_based.sac import SAC
from pyrado.environment_wrappers.action_normalization import ActNormWrapper, ObsActCatWrapper
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# parser.add_argument('--env-name', default="Pendulum-v0_sim2real", #"AcrobotEnv-v2_euler_bsz512",
# parser.add_argument('--env-name', default="Acrobot-v0_gradvi", #"AcrobotEnv-v2_euler_bsz512",
# parser.add_argument('--env-name', default="AcrobotEnv-v2_tanheval_init_dt005_on_policy",
# parser.add_argument('--env-name', default="Acrobot-v0_explore",
parser.add_argument('--env-name', default="Cartpole-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=128, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--grad_vi', action="store_true",
                    help='also perform value gradient iterations')
parser.add_argument('--explore', action="store_true",
                    help='explore using value gradient and value hessian')
parser.add_argument('--eval_init', action="store_true",
                    help='evaluate from bottom as init state')
parser.add_argument('--exploration', type=str, default='random',
                    help='exploration type : {random, on-policy, backward}')
parser.add_argument('--transfer', action="store_true", 
                    help='sim2sim? (default: False)')
parser.add_argument('--load', action="store_true",
                    help='load checkpoint?')
parser.add_argument('--save', action="store_true",
                    help='save checkpoint?')
parser.add_argument('--test', action="store_true",
                    help='test')
parser.add_argument('--task_dim', type=int, default=64,
                    help='number of dimenstions for task representation')
parser.add_argument('--grad_explicit', action="store_true",
                    help='use explicit grad networks')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make(args.env_name)
# env.seed(args.seed)
# env.action_space.seed(args.seed)

T = 300
dt = 0.05
# if dt==0.05:
#     T = 250
device = torch.device("cuda" if args.cuda else "cpu")
if args.exploration == 'on-policy':
    if 'Acrobot' in args.env_name:
        env = AcrobotEnv(1, device=device, dt=dt, T=T)
        env_tr = AcrobotEnv(1, device=device, dt=dt, T=T, l1=1.2, m1=1.2)
        test_env = AcrobotEnv(64, device=device, dt=dt, T=T)
        test_env_tr = AcrobotEnv(64, device=device, dt=dt, T=T, l1=1.2, m1=1.2)
    # env = SGAcroEnvWrapper(1, device=device)
    elif 'Pendulum' in args.env_name:
        env = PendulumDynamics(1, device, action_clamp=True, l=1.0, m=1.0)
        env_tr = PendulumDynamics(1, device, action_clamp=True, l=1.4, m=1.4)
        test_env = PendulumDynamics(64, device, action_clamp=True, l=1.0, m=1.0)
        test_env_tr = PendulumDynamics(64, device, action_clamp=True, l=1.4, m=1.4)
    else:
        env_hparams_tr = dict(
            dt=1 / 20.0,
            max_steps=300,
            long=True,
            simple_dynamics=False,
            wild_init=True,
        )
        env_hparams = dict(
            dt=1 / 20.0,
            max_steps=300,
            long=True,
            simple_dynamics=True,
            wild_init=True,
        )
        env = QCartPoleSwingUpSim(**env_hparams)
        env_tr = QCartPoleSwingUpSim(**env_hparams_tr)
        test_env = QCartPoleSwingUpSim(**env_hparams)
        test_env_tr = QCartPoleSwingUpSim(**env_hparams_tr)
        # env = ObsVelFiltWrapper(env, idcs_pos=["theta", "alpha"], idcs_vel=["theta_dot", "alpha_dot"])
        env = ActNormWrapper(env)
        env = ObsActCatWrapper(env)

        env_tr = ActNormWrapper(env_tr)
        env_tr = ObsActCatWrapper(env_tr)

        test_env = ActNormWrapper(test_env)
        test_env = ObsActCatWrapper(test_env)

        test_env_tr = ActNormWrapper(test_env_tr)
        test_env_tr = ObsActCatWrapper(test_env_tr)
    args.updates_per_step = 1
    print_interval = 2000
else:
    if 'Acrobot' in args.env_name:
        env = AcrobotEnv(64, device=device, dt=dt, T=T)
        # env_tr = PendulumDynamics(1, device, action_clamp=True, l=0.9, m=0.9)
        # test_env = PendulumDynamics(64, device, action_clamp=True)
        # test_env_tr = PendulumDynamics(64, device, action_clamp=True, l=0.9, m=0.9)
    # env = SGAcroEnvWrapper(64, device=device)
    else:
        env = PendulumDynamics(64, device, action_clamp=True)
    args.updates_per_step = 128
    print_interval = 10

task_vec = torch.zeros(args.task_dim).to(device)
task_vec_var = torch.autograd.Variable(task_vec, requires_grad=True)
task_vec_optim = Adam([task_vec_var,], lr=args.lr)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
# suffix = "sim2real_sac_gradviFalse_lm14_task_vecsplitconst_ilc_action"
# suffix = "acrobot_gravi_scratch_onpolicy_gradvi_stateaction_coeff2001acoeff11"
# suffix = "acrobot_gravi_transfer1212_offpolicy_gradvi_stateaction_coeff2000acoeff12"
# suffix = "acrobot_gravi_scratch_offpolicy_gradvi_stateaction_coeff2000acoeff12_dt001_again"
suffix = "offpolicy_stateaction_coeff22"
if "Cartpole" in args.env_name:
    agent = SAC(env.obs_space.shape[0], env.act_space, args, env, args.task_dim, task_vec_optim, task_vec_var)
else:    
    agent = SAC(env.observation_space.shape[0], env.action_space, args, env, args.task_dim, task_vec_optim, task_vec_var)
if args.load:
    # agent.load_checkpoint("checkpoints/sac_checkpoint_Pendulum-v0_sim2real_sim2real_sac")
    # agent.load_checkpoint("checkpoints/sac_checkpoint_Pendulum-v0_sim2real_sim2real_sac_gradvi")
    # agent.load_checkpoint("checkpoints/sac_checkpoint_Pendulum-v0_sim2real_sim2real_sac_gradvi_lm14_task_vec")
    # agent.load_checkpoint("checkpoints/sac_checkpoint_Acrobot-v0_gradvi_acrobot_gravi(False)")
    agent.load_checkpoint("checkpoints/sac_checkpoint_Acrobot-v0_gradvi_acrobot_gravi")

#Tensorboard
if not args.test:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter('runs/{}_SAC_{}_gradvi({})_expl({})_transfer({})_load({})_dt({})_T({})_{}'.format(
                        args.env_name, suffix, args.grad_vi,args.exploration,args.transfer,
                        args.load, dt, T, timestamp))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)
memory_tr = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
max_reward = -1000
step_count = 0
state = env.reset()
if 'Cartpole' in args.env_name:
    state = torch.Tensor(state).to(device).unsqueeze(0).float()
start_transfer = args.load
off_policy_transfer = True
policy = lambda state, vec: agent.select_action_batch(state, vec)
print_flag= args.load
started_transfer = False
if args.load:
    print_interval = T
for i_episode in itertools.count(1):
    if start_transfer and not started_transfer:
        total_numsteps=0
        updates=0
        max_reward=-1000
        step_count=0
        state=env_tr.reset()
        if 'Cartpole' in args.env_name:
            state = torch.Tensor(state).to(device).unsqueeze(0).float()
        started_transfer=True
    episode_reward = 0
    episode_steps = 0
    done = False
    if step_count==T:
        if start_transfer:
            state = env_tr.reset()
            if 'Cartpole' in args.env_name:
                state = torch.Tensor(state).to(device).unsqueeze(0).float()
        else:
            state = env.reset()
            if 'Cartpole' in args.env_name:
                state = torch.Tensor(state).to(device).unsqueeze(0).float()
        step_count = 0

    # while not done:
        # if args.start_steps > total_numsteps:
        #     action = env.action_space.sample()  # Sample random action
        # else:
    # if args.explore:
    task_vec_var_batch = torch.stack([task_vec_var]*state.shape[0], dim=0)
    if args.exploration == 'backward':
        # if len(memory) < args.batch_size:
        if i_episode <= 2:
            if 'pendulum' in env.env_name:
                state, curv = env.sample_goal_state(radius=[0.1,0.1*8/np.pi])
            else:
                # state, curv = env.sample_goal_state(radius=[0.05,0.05,0.05,0.05])
                state, curv = env.sample_goal_state(radius=[0.005,0.005,0.005,0.005])
        else:
            # state, curv = env.sample_new_states_ilqr_backward(memory.buffer, memory.count[:len(memory)], memory.curv[:len(memory)], memory.density[:len(memory)], agent.vfn)
            # state, curv = env.sample_new_states_ilqr_backward(memory.buffer, memory.count[:len(memory)], memory.curv[:len(memory)], memory.density[:len(memory)], agent.vfn, args, env)
            state, old_state, curv = env.sample_new_states_ilqr_backward_rrt(memory.buffer, memory.count[:len(memory)], memory.curv[:len(memory)], memory.density[:len(memory)], agent.vfn, args, env, memory.states, policy)
        action = agent.select_action_batch(state, task_vec_var_batch)  # Sample action from policy
    elif args.exploration == 'random':
        state = env.reset()
        action = agent.select_action_batch(state, task_vec_var_batch)  # Sample action from policy
    else:
        if args.start_steps > total_numsteps and not start_transfer:
            if 'Cartpole' in args.env_name:
                action = torch.rand((1,1)).to(device)*2-1
            else:
                action = env.action_space.sample().to(device).unsqueeze(-1)
        else:
            action = agent.select_action_batch(state, task_vec_var_batch)  # Sample action from policy

    if len(memory) > args.batch_size and (not start_transfer or not off_policy_transfer):
        # print(len(memory), args.batch_size, updates, i_episode)
        # ipdb.set_trace()

        # Number of updates per step in environment
        for i in range(args.updates_per_step):
            # Update parameters of all the networks
            task_vec_batch = torch.stack([task_vec]*args.batch_size, dim=0)
            env_arg = None
            if start_transfer:
                env_arg = env_tr
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates, task_vec_batch, env_arg)
            if not args.test:
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
            updates += 1

    if len(memory_tr) > T and start_transfer and off_policy_transfer:
        # Number of updates per step in environment
        for i in range(args.updates_per_step):
            # Update parameters of all the networks
            bsz_tr = min(args.batch_size, len(memory_tr))
            task_vec_var_batch = torch.stack([task_vec_var]*bsz_tr, dim=0)
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters_with_real(memory_tr, bsz_tr, updates, task_vec_var_batch)
            if not args.test:
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
            updates += 1

    if args.exploration != 'on-policy':
        if start_transfer:
            env_tr.state = state
        else:
            env.state = state
    state1 = state.detach().clone()
    if 'Cartpole' in args.env_name:
        if start_transfer:
            next_state, reward, done, info = env_tr.step_diff_state(state, action)#, return_costs=True) # Step
        else:
            next_state, reward, done, info = env.step_diff_state(state, action)#, return_costs=True) # Step
        reward = reward.unsqueeze(0)
    else:
        if start_transfer:
            next_state, reward, done, _ = env_tr.step(action)#, return_costs=True) # Step
        else:
            next_state, reward, done, _ = env.step(action)#, return_costs=True) # Step

    while torch.isinf(next_state).sum() or torch.isnan(next_state).sum():
        if start_transfer:
            state = env_tr.reset()
        else:
            state = env.reset()
        print("found inf state : train")
        if step_count==0:
            print("found another inf state: train")
        if 'Cartpole' in args.env_name:
            state = torch.Tensor(state).to(device).unsqueeze(0).float()
            if start_transfer:
                next_state, reward, done, info = env_tr.step_diff_state(state, action)#, return_costs=True) # Step
            else:
                next_state, reward, done, info = env.step_diff_state(state, action)#, return_costs=True) # Step
            reward = reward.unsqueeze(0)
        else:
            if start_transfer:
                next_state, reward, done, _ = env_tr.step(action)#, return_costs=True) # Step
            else:
                next_state, reward, done, _ = env.step(action)#, return_costs=True) # Step
        step_count = 0
    # if torch.isinf(next_state).sum() or torch.isnan(next_state).sum():
    #     ipdb.set_trace()
    # next_state1, reward1, done1 = env1(state[:1],action[:1],return_costs=True) # Step
    # ipdb.set_trace()
    episode_steps += 1
    total_numsteps += 1
    episode_reward = reward

    # Ignore the "done" signal if it comes from hitting the time horizon.
    # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
    # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
    mask = torch.ones_like(action[:, 0])
    # mask = done.float()
    # reward = reward.squeeze(-1)
    # ipdb.set_trace()
    if args.exploration == 'backward':
        memory.push_density(state, action, reward, next_state, mask, curv) # Append transition to memory
    else:    
        if start_transfer and off_policy_transfer:
            memory_tr.push(state, action, reward, next_state, mask, info['th_ddot']) # Append transition to memory
        else:
            memory.push(state, action, reward, next_state, mask, info['th_ddot']) # Append transition to memory

    if False:#done.item():
        if start_transfer:
            state = env_tr.reset()
        else:
            state = env.reset()
        if 'Cartpole' in args.env_name:
            state = torch.Tensor(state).to(device).unsqueeze(0).float()
        step_count = 0
    else:
        state = next_state
        step_count +=1


    if total_numsteps > args.num_steps:
        break

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    # print("Episode: {}".format(i_episode))

    if (i_episode % print_interval == 0 and args.eval is True) or print_flag:
        avg_reward = 0.
        episodes = 10
        states = []
        # for _  in range(episodes):
        if args.eval_init:
            if start_transfer:
                state_eval = env_tr.reset_init()
            else:
                state_eval = env.reset_init()
        else:
            if start_transfer:
                state_eval = test_env_tr.reset()
            else:
                state_eval = test_env.reset()
        if 'Cartpole' in args.env_name:
            state_eval = []
            for i in range(64):
                if start_transfer:
                    state_evali = test_env_tr.reset()
                else:
                    state_evali = test_env.reset()
                state_eval.append(torch.tensor(state_evali).to(device).float())
            state_eval = torch.stack(state_eval, dim=0)
        if args.exploration=='backward':
            state_eval = env.state = samples_from_buffer(memory.buffer, state_eval)
        episode_reward = None
        done = False
        states.append(state_eval)
        # while not done:
        task_vec_var_batch = torch.stack([task_vec_var]*state_eval.shape[0], dim=0)
        inf_states = torch.ones_like(state_eval[:,:1])
        for j in range(T):
            # ipdb.set_trace()
            try:
                action = agent.select_action_batch(state_eval, task_vec_var_batch, evaluate=True)
            except:
                ipdb.set_trace()
            if 'Cartpole' in args.env_name:
                if start_transfer:
                    next_state_test, reward_test, done_test, _ = test_env_tr.step_diff_state(state_eval, action)
                else:
                    next_state_test, reward_test, done_test, _ = test_env.step_diff_state(state_eval, action)
            else:
                if start_transfer:
                    next_state_test, reward_test, done_test, _ = test_env_tr.step(action)
                else:
                    next_state_test, reward_test, done_test, _ = test_env.step(action)
            if episode_reward is None:
                episode_reward = reward_test*0
            episode_reward += reward_test * inf_states.squeeze()
            ipdb.set_trace()
            inf_states = (torch.isnan(next_state_test).sum(dim=-1)==0).float().unsqueeze(-1)*(torch.isinf(next_state_test).sum(dim=-1)==0).float().unsqueeze(-1)*inf_states*(1-done_test.float().unsqueeze(-1))
            if torch.isinf(next_state_test).sum():
                print("found inf_states: test")
                # ipdb.set_trace()
            state_eval = next_state_test
            state_eval[inf_states[:,0]==0]=0
            states.append(state_eval)

        avg_reward = episode_reward.mean()
        print_flag=False
        if avg_reward > -3 and 'Pendulum' in args.env_name and not start_transfer and args.transfer:
            start_transfer=True
            print_interval=T
            print_flag=True
            print(avg_reward, "inside")
            ipdb.set_trace()

            # ipdb.set_trace()
        if avg_reward>max_reward and args.save and not args.test:
            max_reward = avg_reward
            states = torch.stack(states)
            # suffix = "rrt_lqr_act_cost005_rand200_top20_step13_5200switchonpolicy"
            # agent.save_checkpoint(args.env_name, suffix)
            agent.save_checkpoint(args.env_name, ckpt_path='runs/{}_SAC_{}_gradvi({})_expl({})_transfer({})_load({})_dt({})_T({})_{}/best_checkpoint_transfer({}).ckpt'.format(
                        args.env_name, suffix, args.grad_vi,args.exploration,args.transfer,
                        args.load, dt, T, timestamp, start_transfer))
            # np.save("checkpoints/states_{}_{}.npy".format(args.env_name, suffix), states)
            np.save('runs/{}_SAC_{}_gradvi({})_expl({})_transfer({})_load({})_dt({})_T({})_{}/best_states_transfer({}).npy'.format(
                        args.env_name, suffix, args.grad_vi,args.exploration,args.transfer,
                        args.load, dt, T, timestamp, start_transfer), states)
            print('saved states with rewards : ', avg_reward)
        if avg_reward > 265 and 'Cartpole' in args.env_name and not start_transfer and args.transfer:
            start_transfer=True
            print_interval=T
            print_flag=True
            print(avg_reward, "finished and transfering")
            max_reward = -1000
        if args.exploration == 'backward' and not args.test:
            env.plot_samples(memory.buffer, agent.vfn, i_episode, old_state, state1)
        # avg_reward /= episodes
        # ipdb.set_trace()
        if not args.test:
            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}, Episode {}".format(episodes, avg_reward, i_episode))
        print("----------------------------------------")

env.close()

