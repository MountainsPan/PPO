import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions import Categorical
import gym
import time
import numpy as np


class Memory:
    def __init__(self):
        self.actions = []  # 行动(共4种)
        self.states = []  # 状态, 由8个数字组成
        self.log_probs = []  # 概率
        self.rewards = []  # 奖励
        self.is_dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_dones[:]

    def all_append(self, one_action, one_state, one_log_prob, one_reward, one_is_dones):
        self.actions.append(one_action)
        self.states.append(one_state)
        self.log_probs.append(one_log_prob)
        self.rewards.append(one_reward)
        self.is_dones.append(one_is_dones)


# class ResBlock(nn.Module):
#     def __init__(self, n_chans):
#         super().__init__()
#         self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
#         self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
#         torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
#         torch.nn.init.constant_(self.batch_norm.weight, 0.5)
#         torch.nn.init.zeros_(self.batch_norm.bias)
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.batch_norm(out)
#         out = torch.relu(out)
#         return out + x


class Action(torch.nn.Module):
    def __init__(self, state_dim=8, action_dim=4, n_blocks=10):
        super().__init__()
        # actor

        self.fc1 = torch.nn.Linear(state_dim, 128)
        # self.stacks = nn.Sequential(
        #     *(n_blocks * [ResBlock(n_chans=16)])
        # )
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, action_dim)

    def forward(self, k):
        k = torch.relu(self.fc1(k))
        # k = self.stacks(k)
        k = torch.relu(self.fc2(k))
        k = self.fc3(k)
        return torch.softmax(k, dim=-1)


class Value(torch.nn.Module):
    def __init__(self, state_dim=8, n_blocks=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        # self.stacks = nn.Sequential(
        #     *(n_blocks * [ResBlock(n_chans=16)])
        # )
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, l):
        l = torch.relu(self.fc1(l))
        # l = self.stacks(l)
        l = torch.relu(self.fc2(l))
        l = self.fc3(l)
        return l


class PPOAgent:
    def __init__(self, action_net_in, value_net_in, lr, betas, gamma, K_epochs):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_layer = action_net_in
        self.value_layer = value_net_in

        self.optimizer = torch.optim.Adam(
            [{"params": self.action_layer.parameters()}, {"params": self.value_layer.parameters()}], lr=lr, betas=betas)

        self.MseLoss = torch.nn.MSELoss()

    def evaluate(self, state, action):

        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)

        # 返回行动概率密度, 评判值, 行动概率熵
        return action_log_probs, torch.squeeze(state_value), dist_entropy

    def update(self, memory):
        # 预测状态回报
        returns = []
        each_reward = 0
        for reward, is_done in zip(reversed(memory.rewards), reversed(memory.is_dones)):
            # 回合结束
            if is_done:
                each_reward = 0

            each_reward = reward + (self.gamma * each_reward)

            returns.insert(0, each_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  #

        old_states = torch.tensor(memory.states)
        old_actions = torch.tensor(memory.actions)
        old_log_probs = torch.tensor(memory.log_probs)

        #代优化 K 次:
        for _ in range(5):
            # Evaluating old actions and values : 新策略 重用 旧样本进行训练
            log_probs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            ratios = torch.exp(log_probs - old_log_probs.detach())

            advantages = returns - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def act(self, state):

        state = torch.from_numpy(state).float()
        # 计算4个方向概率
        action_probs = self.action_layer(state)
        # 通过最大概率计算最终行动方向
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)


lr = 0.002  # learning rate
betas = (0.9, 0.999)
gamma = 0.99  # discount factor
K_epochs = 5  # policy迭代更新次数
eps_clip = 0.2  # clip parameter for PPO  论文中表明0.2效果不错

action_net_cin = Action()
value_net_cin = Value()

agent = PPOAgent(action_net_cin, value_net_cin, lr, betas, gamma, K_epochs)
memory = Memory()
# agent.network.train()  # Switch network into training mode
EPISODE_PER_BATCH = 5  # update the  agent every 5 episode

env = gym.make('LunarLander-v2', render_mode='rgb_array')
rewards_list = []
for i in range(200):

    rewards = []

    for episode in range(EPISODE_PER_BATCH):

        state = env.reset()[0]

        while True:

            with torch.no_grad():
                action, action_prob = agent.act(state)

            next_state, reward, done, _, _ = env.step(action)

            rewards.append(reward)

            memory.all_append(action, state, action_prob, reward, done)
            state = next_state

            if len(memory.rewards) >= 1200:
                agent.update(memory)
                memory.clear_memory()

            if done or len(rewards) > 1024:
                rewards_list.append(np.sum(rewards))

                break
    print(f"epoch: {i} ,rewards looks like ", rewards_list[-1])

plt.plot(range(len(rewards_list)), rewards_list)
plt.show()
plt.close()
env = gym.make('LunarLander-v2', render_mode='human')

while True:
    state = env.reset()[0]
    step = 0
    while True:
        step += 1

        action, action_prob = agent.act(state)
        # agent与环境进行一步交互
        state, reward, terminated, truncated, info = env.step(action)

        if terminated or step >= 600:
            print('游戏结束')
            break
        time.sleep(0.01)
