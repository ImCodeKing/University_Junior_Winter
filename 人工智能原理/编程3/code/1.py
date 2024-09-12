import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
    from Tkinter import PhotoImage
else:
    import tkinter as tk
    from tkinter import PhotoImage

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import random
from tqdm import tqdm


UNIT = 100   # 迷宫中每个格子的像素大小
MAZE_H = 6  # 迷宫的高度（格子数）
MAZE_W = 6  # 迷宫的宽度（格子数）


class ReplayBuffer:
    def __init__(self, capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)

    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 目前队列长度
    def size(self):
        return len(self.buffer)


class Net(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN:
    def __init__(self, n_states, n_hidden, n_actions, learning_rate, gamma, epsilon, target_update, device):
        # 属性分配
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device

        self.count = 0

        # 构建2个神经网络，相同的结构，不同的参数
        self.eval_net = Net(self.n_states, self.n_hidden, self.n_actions).to(self.device)
        self.target_net = Net(self.n_states, self.n_hidden, self.n_actions).to(self.device)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if np.random.random() < self.epsilon:
            with torch.no_grad():
                actions_value = self.eval_net.forward(state)
                action = actions_value.argmax().item()
        else:
            action = np.random.randint(self.n_actions)
        return action

    def train(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_eval = self.eval_net(states).gather(1, actions)  # [b,1]
        q_next = self.target_net(next_states).detach()
        max_next_q_values = q_next.max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # print(q_eval, q_targets)

        dqn_loss = F.mse_loss(q_eval, q_targets)
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        net_renew = False
        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            net_renew = True

        self.count += 1
        return dqn_loss, net_renew


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r'] # 决策空间
        self.n_actions = len(self.action_space)
        self.title('Q-learning')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        """
        迷宫初始化
        """
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([UNIT/2, UNIT/2])
        
        self.bm_stone = PhotoImage(file="obstacles.png")
        self.stone1 = self.canvas.create_image(origin[0]+UNIT * 4, origin[1]+UNIT,image=self.bm_stone)
        self.stone2 = self.canvas.create_image(origin[0]+UNIT, origin[1]+UNIT * 4,image=self.bm_stone)
        self.stone3 = self.canvas.create_image(origin[0]+UNIT*4, origin[1]+UNIT * 3,image=self.bm_stone)
        self.stone4 = self.canvas.create_image(origin[0]+UNIT*3, origin[1]+UNIT * 4,image=self.bm_stone)

        self.bm_yoki = PhotoImage(file="character.png")
        self.yoki = self.canvas.create_image(origin[0], origin[1],image=self.bm_yoki)

        self.bm_Candy = PhotoImage(file="candy.png")
        self.Candy = self.canvas.create_image(origin[0]+4*UNIT, origin[1]+4*UNIT,image=self.bm_Candy)

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.yoki)
        origin = np.array([UNIT/2, UNIT/2])
        
        self.yoki = self.canvas.create_image(origin[0], origin[1], image=self.bm_yoki)
        return self.canvas.coords(self.yoki)

    def step(self, action):
        reward = 0
        s = self.canvas.coords(self.yoki)
        base_action = np.array([0, 0])
        if action == 0:   # 向上移动
            if s[1] > UNIT:
                base_action[1] -= UNIT
                reward += 1
        elif action == 1:   # 向下移动
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
                reward += 1
        elif action == 2:   # 向右移动
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
                reward += 1
        elif action == 3:   # 向左移动
            if s[0] > UNIT:
                base_action[0] -= UNIT
                reward += 1

        self.canvas.move(self.yoki, base_action[0], base_action[1]) 
        s_ = self.canvas.coords(self.yoki)

        distance_punishment = abs(self.canvas.coords(self.Candy)[0] - s_[0]) + abs(self.canvas.coords(self.Candy)[1] - s_[1])
        # 回报函数
        if s_ == self.canvas.coords(self.Candy):
            reward += 100
            done = True
            s_ = [0.0, 0.0]
        elif s_ in [self.canvas.coords(self.stone1), self.canvas.coords(self.stone2),self.canvas.coords(self.stone3),self.canvas.coords(self.stone4)]:
            reward += -100
            done = True
            s_ = [0.0, 0.0]
        else:
            reward += 0
            done = False

        reward += -0.005 * distance_punishment
        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    episodes = 500
    max_steps = 50

    n_states = 2
    n_actions = env.n_actions
    n_hidden = 128

    capacity = 500  # 经验池容量
    lr = 2e-3
    gamma = 0.9

    initial_epsilon = 0.1
    final_epsilon = 0.9
    epsilon_increment = 0.005

    target_update = 200  # 目标网络的参数的更新频率
    batch_size = 32
    min_size = 200  # 经验池超过200后再训练
    loss = np.inf
    renew = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    replay_buffer = ReplayBuffer(capacity)
    agent = DQN(n_states=n_states, n_hidden=n_hidden, n_actions=n_actions, learning_rate=lr, gamma=gamma, epsilon=initial_epsilon,
                target_update=target_update, device=device)

    for episode in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset()
        done = False
        total_reward = 0
        renew_record = False

        with tqdm(range(max_steps), desc="Env Steps", leave=False) as pbar:
            for _ in pbar:
                env.render()
                if done:
                    break

                action = agent.take_action(state)

                next_state, reward, done = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if replay_buffer.size() > min_size:
                    # 从经验池中随机抽样作为训练集
                    s, a, r, ns, d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': s,
                        'actions': a,
                        'next_states': ns,
                        'rewards': r,
                        'dones': d,
                    }
                    loss, renew = agent.train(transition_dict)
                    if renew:
                        renew_record = True

            tqdm.write(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss}, Renew: {renew_record}, Epsilon: {agent.epsilon}")
        agent.epsilon = min(final_epsilon, agent.epsilon + epsilon_increment)

    torch.save(agent.eval_net.state_dict(), 'dqn_eval_net.pth')


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
