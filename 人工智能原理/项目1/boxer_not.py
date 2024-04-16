import time

import pygame, sys, os
from pygame.locals import *
import heapq

from collections import deque

import random


def to_box(level, index):
    if level[index] == '-' or level[index] == '@':
        level[index] = '$'
    else:
        level[index] = '*'


def to_man(level, i):
    if level[i] == '-' or level[i] == '$':
        level[i] = '@'
    else:
        level[i] = '+'


def to_floor(level, i):
    if level[i] == '@' or level[i] == '$':
        level[i] = '-'
    else:
        level[i] = '.'


def to_offset(d, width):
    d4 = [-1, -width, 1, width]
    m4 = ['l', 'u', 'r', 'd']
    return d4[m4.index(d.lower())]


def b_manto(level, width, b, m, t):
    maze = list(level)
    maze[b] = '#'
    if m == t:
        return 1
    queue = deque([])
    queue.append(m)
    d4 = [-1, -width, 1, width]
    m4 = ['l', 'u', 'r', 'd']
    while len(queue) > 0:
        pos = queue.popleft()
        for i in range(4):
            newpos = pos + d4[i]
            if maze[newpos] in ['-', '.']:
                if newpos == t:
                    return 1
                maze[newpos] = i
                queue.append(newpos)
    return 0


def b_manto_2(level, width, b, m, t):
    maze = list(level)
    maze[b] = '#'
    maze[m] = '@'
    if m == t:
        return []
    queue = deque([])
    queue.append(m)
    d4 = [-1, -width, 1, width]
    m4 = ['l', 'u', 'r', 'd']
    while len(queue) > 0:
        pos = queue.popleft()
        for i in range(4):
            newpos = pos + d4[i]
            if maze[newpos] in ['-', '.']:
                maze[newpos] = i
                queue.append(newpos)
                if newpos == t:
                    path = []
                    while maze[t] != '@':
                        path.append(m4[maze[t]])
                        t = t - d4[maze[t]]
                    return path

    return []

map_list = ['------------------------############-------####---#####-------####---#####-------##@$---#####-------####.#######-------#########---------------------------------------------------------------------------------',
            '----#####--------------#---#--------------#$--#------------###--$##-----------#------#---------###-#-##-#---#######---#$##-#####---.##----------------.######-###-#@##---.#----#-----#########----#######--------',
            '####################----#---#---#----##--$.#---#-@-#----##----#---#---#----#####-#---------.--##-----------------##----#---------#-.##--###------##-#####--$-#-------$----##----#-----####################-------']

man_list = [77, 163, 49]

class BoxGame:
    def __init__(self):
        random_map_num = random.randint(0, 2)
        self.level = list(map_list[random_map_num])

        # self.level = list(
        #     '----#####--------------#---#--------------#---#------------###---##-----------#--$---#---------###-#-##-#---#######---#-##-#####---.##-----------------######-###-#@##----#----#-----#########----#######--------')

        self.w = 19
        self.h = 11
        self.man = man_list[random_map_num]
        self.hint = list(self.level)
        self.box_count = self.level.count('$')
        self.solution = []
        self.push = 0
        self.todo = []
        self.auto = 0
        self.sbox = 0
        self.queue = []

    def draw(self, screen, skin):
        w = skin.get_width() / 4
        offset = (w - 4) / 2
        for i in range(0, self.w):
            for j in range(0, self.h):
                # 墙
                if self.level[j * self.w + i] == '#':
                    screen.blit(skin, (i * w, j * w), (0, 2 * w, w, w))
                # 路
                elif self.level[j * self.w + i] == '-':
                    screen.blit(skin, (i * w, j * w), (0, 0, w, w))
                # 人
                elif self.level[j * self.w + i] == '@':
                    screen.blit(skin, (i * w, j * w), (w, 0, w, w))
                # 箱子
                elif self.level[j * self.w + i] == '$':
                    screen.blit(skin, (i * w, j * w), (2 * w, 0, w, w))
                # 目标
                elif self.level[j * self.w + i] == '.':
                    screen.blit(skin, (i * w, j * w), (0, w, w, w))
                # 人在目标上
                elif self.level[j * self.w + i] == '+':
                    screen.blit(skin, (i * w, j * w), (w, w, w, w))
                # 箱子在目标上
                elif self.level[j * self.w + i] == '*':
                    screen.blit(skin, (i * w, j * w), (2 * w, w, w, w))

                if self.sbox != 0 and self.hint[j * self.w + i] == '1':
                    screen.blit(skin, (i * w + offset, j * w + offset), (3 * w, 3 * w, 4, 4))

    def move(self, d):
        self._move(d)
        self.todo = []

    def _move(self, d):
        self.sbox = 0
        h = to_offset(d, self.w)
        h2 = 2 * h
        if self.level[self.man + h] == '-' or self.level[self.man + h] == '.':
            # move
            to_man(self.level, self.man + h)
            to_floor(self.level, self.man)
            self.man += h
            self.solution += d
        elif self.level[self.man + h] == '*' or self.level[self.man + h] == '$' or self.level[self.man + h].isdigit():
            if self.level[self.man + h2] == '-' or self.level[self.man + h2] == '.':
                # push
                to_box(self.level, self.man + h2)
                to_man(self.level, self.man + h)
                to_floor(self.level, self.man)
                self.man += h
                self.solution += d.upper()
                self.push += 1


    def bfs_search(self):
        visited = set()  # 用于记录已访问过的状态
        queue = deque([(self.level[:], self.man, '')])  # 初始状态加入队列
        directions = {'l': -1, 'u': -self.w, 'r': 1, 'd': self.w}  # 移动方向对应的偏移量

        while queue:
            level, man, path = queue.popleft()  # 取出队列中的状态
            state = (tuple(level), man)  # 将状态转换为不可变的对象，方便存入集合

            if state in visited:  # 如果状态已经访问过，则跳过
                continue

            visited.add(state)  # 将状态标记为已访问

            # 检查是否达到目标状态
            if level.count('*') == self.box_count:
                return path  # 返回移动路径

            # 扩展当前状态
            for direction, offset in directions.items():
                new_man = man + offset  # 计算移动后的人物位置
                if level[new_man] in ('-', '.'):  # 如果移动到空地上
                    new_level = level[:]
                    new_level[man] = '-' if level[man] != '+' else '.'  # 更新人物所在位置
                    new_level[new_man] = '@' if level[new_man] != '.' else '+'  # 更新新位置
                    queue.append((new_level, new_man, path + direction))  # 将新状态加入队列

                elif level[new_man] in ('$', '*'):  # 如果移动到箱子位置
                    new_box = new_man + offset  # 计算箱子推动后的位置
                    if level[new_box] in ('-', '.'):  # 如果箱子推动到空地上
                        new_level = level[:]
                        new_level[man] = '-' if level[man] != '+' else '.'  # 更新人物所在位置
                        new_level[new_man] = '@' if level[new_man] != '.' else '+'  # 更新人物推动后的位置
                        new_level[new_box] = '$' if level[new_box] != '.' else '*'  # 更新箱子推动后的位置
                        queue.append((new_level, new_man, path + direction.upper()))  # 将新状态加入队列

        return None  # 如果搜索失败，则返回空值

    def find_goals(self):
        goals = []
        for i, char in enumerate(self.level):
            if char == '.':
                goals.append((i % self.w, i // self.w))  # 将目标位置的坐标添加到列表中
        return goals

    def heuristic(self, level):
        # 计算所有箱子到目标位置的曼哈顿距离之和作为启发函数值
        total_distance = 0
        goals = self.find_goals()  # 获取所有目标位置的坐标
        for i, char in enumerate(level):
            if char == '$':
                box_x, box_y = i % self.w, i // self.w  # 计算箱子的坐标
                # 计算箱子到所有目标位置的曼哈顿距离，并取最小值
                min_distance = min(abs(box_x - goal[0]) + abs(box_y - goal[1]) for goal in goals)
                total_distance += min_distance
        return total_distance

    def astar_search(self):
        visited = set()  # 用于记录已访问过的状态
        open_list = [(self.heuristic(self.level), self.level[:], self.man, '')]  # 初始状态加入开放列表
        directions = {'l': -1, 'u': -self.w, 'r': 1, 'd': self.w}  # 移动方向对应的偏移量
        count = 0

        while open_list:
            print(count)
            count += 1
            _, level, man, path = heapq.heappop(open_list)  # 从开放列表中取出启发值最小的状态
            state = (tuple(level), man)  # 将状态转换为不可变的对象，方便存入集合

            if state in visited:  # 如果状态已经访问过，则跳过
                continue

            visited.add(state)  # 将状态标记为已访问

            # 检查是否达到目标状态
            if level.count('*') == self.box_count:
                return path  # 返回移动路径

            # 扩展当前状态
            for direction, offset in directions.items():
                new_man = man + offset  # 计算移动后的人物位置
                if level[new_man] in ('-', '.'):  # 如果移动到空地上
                    new_level = level[:]
                    new_level[man] = '-' if level[man] != '+' else '.'  # 更新人物所在位置
                    new_level[new_man] = '@' if level[new_man] != '.' else '+'  # 更新新位置
                    heapq.heappush(open_list, (len(path) + 1 + self.heuristic(new_level), new_level, new_man, path + direction))  # 将新状态加入开放列表

                elif level[new_man] in ('$', '*'):  # 如果移动到箱子位置
                    new_box = new_man + offset  # 计算箱子推动后的位置
                    if level[new_box] in ('-', '.'):  # 如果箱子推动到空地上
                        new_level = level[:]
                        new_level[man] = '-' if level[man] != '+' else '.'  # 更新人物所在位置
                        new_level[new_man] = '@' if level[new_man] != '.' else '+'  # 更新人物推动后的位置
                        new_level[new_box] = '$' if level[new_box] != '.' else '*'  # 更新箱子推动后的位置
                        heapq.heappush(open_list, (len(path) + 1 + self.heuristic(new_level), new_level, new_man, path + direction.upper()))  # 将新状态加入开放列表

        return None  # 如果搜索失败，则返回空值


def main():
    # start pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 300))

    # load skin
    skinfilename = os.path.join('borgar.png')
    try:
        skin = pygame.image.load(skinfilename)
    except pygame.error as msg:
        print('cannot load skin')
        raise SystemExit(msg)
    skin = skin.convert()

    screen.fill(skin.get_at((0, 0)))
    pygame.display.set_caption('BoxGame.py')

    # create BoxGame object
    boxer = BoxGame()
    boxer.draw(screen, skin)

    #
    clock = pygame.time.Clock()
    pygame.key.set_repeat(200, 50)

    # path = boxer.bfs_search()
    path = boxer.astar_search()
    # 1280727
    # 6441394
    # path = 'ullluuullldDuulldddrRRRRRRRRRRRRlllllllluuulluurDluulDDDDDuulldddrRRRRRRRRRRRdrUlllllllluuuLLulDDDuulldddrRRRRRRRRRRRuRDldR'
    # path = 'dddllllllldlldddrruLdlUUUluRRRuulluRdrddRRRRRRRRRRdddLLLLdlUUUUluRRRRRRdRRurD'
    print(path)

    for mv in path:
        clock.tick(60)

        boxer.move(mv)
        boxer.draw(screen, skin)

        pygame.display.update()
        pygame.display.set_caption(boxer.solution.__len__().__str__() + '/' + boxer.push.__str__() + ' - boxer.py')
        time.sleep(0.2)

    # main game loop
    while True:
        clock.tick(60)

        if boxer.auto == 0:
            for event in pygame.event.get():
                if event.type == QUIT:
                    # print boxer.solution
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_LEFT:
                        boxer.move('l')
                        boxer.draw(screen, skin)
                    elif event.key == K_UP:
                        boxer.move('u')
                        boxer.draw(screen, skin)
                    elif event.key == K_RIGHT:
                        boxer.move('r')
                        boxer.draw(screen, skin)
                    elif event.key == K_DOWN:
                        boxer.move('d')
                        boxer.draw(screen, skin)
        else:
            boxer.automove()
            boxer.draw(screen, skin)

        pygame.display.update()
        pygame.display.set_caption(boxer.solution.__len__().__str__() + '/' + boxer.push.__str__() + ' - boxer.py')


if __name__ == '__main__':
    main()
