import numpy as np
import matplotlib.pyplot as plt
import ipdb
# T = 401
# dt = 0.05
# states = np.load('checkpoints/states_AcrobotEnv-v2_tanheval_init_dt005_on_policy_.npy')
# plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,0], label='th1')
# plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,1], label='th2')
# plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,2], label='th1d')
# plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,3], label='th2d')
# plt.legend()
# plt.savefig('checkpoints/states_plot_Acrobot_v2_tanheval_init_dt005_on_policy.png')
# plt.clf()


# T = 401
# dt = 0.01
# states = np.load('checkpoints/states_AcrobotEnv-v2_tanheval_init_dt001_.npy')
# ipdb.set_trace()
# plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,0], label='th1')
# plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,1], label='th2')
# plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,2], label='th1d')
# plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,3], label='th2d')
# plt.legend()
# plt.savefig('checkpoints/states_plot_AcrobotEnv-v2_tanheval_init_dt001.png')


T = 251
dt = 0.05
states = np.load('checkpoints/states_Acrobot-v0_gradvi_acrobot_gravi.npy')[:,1]
	# states_Acrobot-v0_explore_rrt_lqr_act_cost005_rand200_top20_step13_5200switchonpolicy.npy')[:,1]
ipdb.set_trace()
plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,0], label='th1')
plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,1], label='th2')
plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,2], label='th1d')
plt.plot(np.arange(T)*dt, states.reshape(-1, 1, 4)[:,0,3], label='th2d')
plt.legend()
plt.savefig('checkpoints/states_Acrobot-v0_gradvi_acrobot_gravi.png')
	# states_Acrobot-v0_explore_rrt_lqr_act_cost005_rand200_top20_step13_5200switchonpolicy.png')


