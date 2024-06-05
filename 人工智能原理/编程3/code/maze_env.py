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
from collections import deque
import random
from tqdm import tqdm


UNIT = 100   # 迷宫中每个格子的像素大小
MAZE_H = 6  # 迷宫的高度（格子数）
MAZE_W = 6  # 迷宫的宽度（格子数）


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


def train_dqn(model, optimizer, criterion, replay_buffer, batch_size, gamma):
    if replay_buffer.size() < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    current_q_values = model(states).gather(1, actions)
    max_next_q_values = model(next_states).max(1)[0].unsqueeze(1)
    expected_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    loss = criterion(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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
        
        self.yoki = self.canvas.create_image(origin[0], origin[1],image=self.bm_yoki)
        return self.canvas.coords(self.yoki)

    def step(self, action):
        s = self.canvas.coords(self.yoki)
        base_action = np.array([0, 0])
        if action == 0:   # 向上移动
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # 向下移动
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # 向右移动
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # 向左移动
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.yoki, base_action[0], base_action[1]) 
        s_ = self.canvas.coords(self.yoki)

        # 回报函数
        if s_ == self.canvas.coords(self.Candy):
            reward = 1
            done = True
            s_ = [-100.0, -100.0]
        elif s_ in [self.canvas.coords(self.stone1), self.canvas.coords(self.stone2),self.canvas.coords(self.stone3),self.canvas.coords(self.stone4)]:
            reward = -1
            done = True
            s_ = [-100.0, -100.0]
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    state_dim = 2
    action_dim = env.n_actions

    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(10000)

    batch_size = 64
    gamma = 0.99
    episodes = 500
    epsilon_decay = 0.995
    min_epsilon = 0.01
    epsilon = 1.0
    max_steps = 50

    for episode in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset()
        done = False
        total_reward = 0

        with tqdm(range(max_steps), desc="Env Steps", leave=False) as pbar:
            for step in pbar:
                env.render()
                if done:
                    break

                if random.random() < epsilon:
                    action = random.randint(0, env.n_actions - 1)
                else:
                    with torch.no_grad():
                        action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()

                next_state, reward, done = env.step(action)
                replay_buffer.add((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                train_dqn(model, optimizer, criterion, replay_buffer, batch_size, gamma)

            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            tqdm.write(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
