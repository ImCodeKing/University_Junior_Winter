import pygame, sys, os
from pygame.locals import *
import heapq

from collections import deque


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

def to_box_121(level, i, ori_idx, box_count):
    if level[i] == '-' or level[i] == '@':
        if level[ori_idx].isdigit():
            level[i] = level[ori_idx]
        elif not 0 <= ord(level[i]) - ord('A') < box_count:
            box = (ord(level[ori_idx]) - ord('A') - box_count) % 10
            level[i] = str(box)
        else:
            level[i] = '$'
    else:
        if 0 <= ord(level[i]) - ord('A') < box_count:
            if ord(level[ori_idx]) - ord('A') < box_count:
                if not (str(ord(level[i]) - ord('A')) == level[ori_idx]):
                    # 10 * 目标编号 + 箱子编号 + box_count（防止冲突）
                    level[i] = chr((box_count + int(level[ori_idx]) + 10 * (ord(level[i]) - ord('A'))) + ord('A'))
            else:
                box = (ord(level[ori_idx]) - ord('A') - box_count) % 10
                if not (ord(level[i]) - ord('A') == box):
                    level[i] = chr((box_count + box + 10 * (ord(level[i]) - ord('A'))) + ord('A'))
        else:
            level[i] = '*'


def to_man_121(level, i, box_count):
    if level[i] == '-' or level[i] == '$' or (level[i].isdigit() and 0 <= int(level[i]) < box_count):
        level[i] = '@'
    else:
        if 0 <= ord(level[i]) - ord('A') < box_count:
            level[i] = chr(ord(level[i]) - ord('A') + ord('a'))
        elif ord(level[i]) - box_count - ord('A') >= 0:
            tar = (ord(level[i]) - ord('A') - box_count) // 10
            level[i] = chr(tar + ord('a'))
        else:
            level[i] = '+'


def to_floor_121(level, i, box_count):
    if level[i] == '@' or level[i] == '$':
        level[i] = '-'
    else:
        if 0 <= ord(level[i]) - ord('a') < box_count:
            level[i] = chr(ord(level[i]) - ord('a') + ord('A'))
        else:
            level[i] = '.'


def to_offset(d, width):
    d4 = [-1, -width, 1, width]
    m4 = ['l', 'u', 'r', 'd']
    return d4[m4.index(d.lower())]


def trans_mix(box, tar):
    if box == '0' and tar == 'A':
        return 'D'
    elif box == '0' and tar == 'B':
        return 'E'
    elif box == '0' and tar == 'C':
        return 'F'
    elif box == '1' and tar == 'A':
        return 'G'
    elif box == '1' and tar == 'B':
        return 'H'
    elif box == '1' and tar == 'C':
        return 'I'
    elif box == '2' and tar == 'A':
        return 'J'
    elif box == '2' and tar == 'B':
        return 'K'
    elif box == '2' and tar == 'C':
        return 'L'


def get_ori(char):
    #    |A B C
    # ---|------
    # 0  |D E F
    # 1  |G H I
    # 2  |J K L
    if char == 'D':
        return '0', 'A'
    elif char == 'E':
        return '0', 'B'
    elif char == 'F':
        return '0', 'C'
    elif char == 'G':
        return '1', 'A'
    elif char == 'H':
        return '1', 'B'
    elif char == 'I':
        return '1', 'C'
    elif char == 'J':
        return '2', 'A'
    elif char == 'K':
        return '2', 'B'
    elif char == 'L':
        return '2', 'C'


map_list = ['------------------------############-------####---#####-------####---#####-------##@$---#####-------####.#######-------#########---------------------------------------------------------------------------------',
            '----#####--------------#---#--------------#$--#------------###--$##-----------#------#---------###-#-##-#---#######---#$##-#####---.##----------------.######-###-#@##---.#----#-----#########----#######--------',
            '####################----#---#---#----##--$.#---#-@-#----##----#---#---#----#####-#---------.--##-----------------##----#---------#-.##--###------##-#####--$-#-------$----##----#-----####################-------',
            '----#####--------------#---#--------------#---#------------###---##-----------#------#---------###-#-##-#---#######---#-##-#####---C##-1--2----------B-######-###-#@##--0A#----#-----#########----#######--------']


man_list = [83, 163, 49, 163]

path_list = ['RRurrdLulD',
             'ullluuullldDuulldddrRRRRRRRRRRRRlllllllluuulluurDluulDDDDDuulldddrRRRRRRRRRRRdrUlllllllluuuLLulDDDuulldddrRRRRRRRRRRRuRDldR',
             'dddllllllldlldddrruLdlUUUluRRRuulluRdrddRRRRRRRRRRdddLLLLdlUUUUluRRRRRRdRRurD',
             'ulllllLuuulldddRRRRRRRRRRRdrRlUllllllllllllulldRRRRRRRRRRRRRRluRR']

class BoxGame:
    def __init__(self, mode):
        self.level = list(map_list[mode])
        self.man = man_list[mode]
        # self.level = list(
        #     '----#####--------------#---#--------------#---#------------###---##-----------#--$---#---------###-#-##-#---#######---#-##-#####---.##-----------------######-###-#@##----#----#-----#########----#######--------')

        self.w = 19
        self.h = 11
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

class BoxGame121:
    def __init__(self, mode):
        self.level = list(map_list[mode])
        self.man = man_list[mode]

        self.box_count = 3
        # self.level = list(
        #     '----#####--------------#---#--------------#---#------------###---##-----------#------#---------###-#-##-#---#######---#-##-#####---C##-1--2----------B-######-###-#@##--0A#----#-----#########----#######--------')

        self.w = 19
        self.h = 11
        self.hint = list(self.level)
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

                else:
                    if self.level[j * self.w + i].isdigit():
                        screen.blit(skin, (i * w, j * w), (2 * w, 0, w, w))
                        font = pygame.font.Font(None, 40)
                        text = font.render(self.level[j * self.w + i], True, (0, 0, 0))  # 将标号转换为文本
                        text_rect = text.get_rect(center=(i * w + w // 2, j * w + w // 2))  # 文本显示在箱子中心位置
                        screen.blit(text, text_rect)

                    elif ord(self.level[j * self.w + i]) - ord('A') >= self.box_count and ord(self.level[j * self.w + i]) - ord('a') < 0:
                        box = (ord(self.level[j * self.w + i]) - ord('A') - self.box_count) % 10
                        tar = (ord(self.level[j * self.w + i]) - ord('A') - self.box_count) // 10

                        screen.blit(skin, (i * w, j * w), (2 * w, 0, w, w))
                        font = pygame.font.Font(None, 25)

                        text_box = font.render(str(box), True, (0, 0, 0))  # 将标号转换为文本
                        text_rect_box = text_box.get_rect(center=(i * w + w // 1.2, j * w + w // 2))  # 文本显示在箱子中心位置
                        screen.blit(text_box, text_rect_box)

                        text_tar = font.render(str(tar), True, (0, 0, 255))  # 将标号转换为文本
                        text_rect_tar = text_tar.get_rect(center=(i * w + w // 4, j * w + w // 2))  # 文本显示在箱子中心位置
                        screen.blit(text_tar, text_rect_tar)

                    else:
                        if 0 <= ord(self.level[j * self.w + i]) - ord('A') < self.box_count:
                            screen.blit(skin, (i * w, j * w), (0, w, w, w))
                            font = pygame.font.Font(None, 25)
                            text = font.render(str(ord(self.level[j * self.w + i]) - ord('A')), True, (0, 0, 0))  # 将标号转换为文本
                            text_rect = text.get_rect(center=(i * w + w // 2, j * w + w // 2))  # 文本显示在箱子中心位置
                            screen.blit(text, text_rect)
                        else:
                            screen.blit(skin, (i * w, j * w), (w, 0, w, w))
                            font = pygame.font.Font(None, 25)
                            text = font.render(str(ord(self.level[j * self.w + i]) - ord('a')), True, (0, 0, 0))  # 将标号转换为文本
                            text_rect = text.get_rect(center=(i * w + w // 2, j * w + w // 2))  # 文本显示在箱子中心位置
                            screen.blit(text, text_rect)

                if self.sbox != 0 and self.hint[j * self.w + i] == '1':
                    screen.blit(skin, (i * w + offset, j * w + offset), (3 * w, 3 * w, 4, 4))

    def move(self, d):
        self._move(d)
        self.todo = []

    def _move(self, d):
        self.sbox = 0
        h = to_offset(d, self.w)
        h2 = 2 * h
        if self.level[self.man + h] == '-' or self.level[self.man + h] == '.' or 0 <= ord(self.level[self.man + h]) - ord('A') < self.box_count:
            # move
            to_man_121(self.level, self.man + h, self.box_count)
            to_floor_121(self.level, self.man, self.box_count)
            self.man += h
            self.solution += d
        elif self.level[self.man + h] == '*' or self.level[self.man + h] == '$' or self.level[self.man + h].isdigit() or ord(self.level[self.man + h]) - ord('A') >= self.box_count:
            if self.level[self.man + h2] == '-' or self.level[self.man + h2] == '.' or 0 <= ord(self.level[self.man + h2]) - ord('A') < self.box_count:
                # push
                to_box_121(self.level, self.man + h2, self.man + h, self.box_count)
                to_man_121(self.level, self.man + h, self.box_count)
                to_floor_121(self.level, self.man, self.box_count)
                self.man += h
                self.solution += d.upper()
                self.push += 1

    def bfs_search_one2one(self):
        visited = set()  # 用于记录已访问过的状态
        queue = deque([(self.level[:], self.man, '')])  # 初始状态加入队列
        directions = {'l': -1, 'u': -self.w, 'r': 1, 'd': self.w}  # 移动方向对应的偏移量
        count = 0

        while queue:
            print(count)
            count += 1
            level, man, path = queue.popleft()  # 取出队列中的状态
            state = (tuple(level), man)  # 将状态转换为不可变的对象，方便存入集合

            if state in visited:  # 如果状态已经访问过，则跳过
                continue

            visited.add(state)  # 将状态标记为已访问

            # 检查是否达到目标状态
            if self.done_check(level):
                return path  # 返回移动路径

            # 扩展当前状态
            for direction, offset in directions.items():
                new_man = man + offset  # 计算移动后的人物位置
                if level[new_man] in ('-', 'A', 'B', 'C'):  # 如果移动到空地上
                    new_level = level[:]
                    new_level[man] = '-' if level[man] not in ('a', 'b', 'c') else chr(ord(level[man]) - ord('a') + ord('A'))  # 更新人物所在位置
                    new_level[new_man] = '@' if level[man] not in ('A', 'B', 'C') else chr(ord(level[man]) - ord('A') + ord('a'))  # 更新新位置
                    queue.append((new_level, new_man, path + direction))  # 将新状态加入队列

                elif level[new_man] in ('0', '1', '2', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'k', 'L'):  # 如果移动到箱子位置
                    new_box = new_man + offset  # 计算箱子推动后的位置
                    if level[new_box] in ('-', 'A', 'B', 'C'):  # 如果箱子推动到空地上
                        new_level = level[:]
                        if level[man] not in ('a', 'b', 'c'):
                            new_level[man] = '-'   # 更新人物所在位置
                        else:
                            new_level[man] = chr(ord(level[man]) - ord('a') + ord('A'))

                        if level[new_man] not in ('D', 'E', 'F', 'G', 'H', 'I', 'J', 'k', 'L'):
                            new_level[new_man] = '@'   # 更新人物推动后的位置
                        else:
                            _, tar = get_ori(level[new_man])
                            new_level[new_man] = chr(ord(tar) - ord('A') + ord('a'))

                        if level[new_box] not in ('A', 'B', 'C'):
                            if level[new_man] not in ('D', 'E', 'F', 'G', 'H', 'I', 'J', 'k', 'L'):
                                new_level[new_box] = level[new_man]  # 更新箱子推动后的位置
                            else:
                                box, _ = get_ori(level[new_man])
                                new_level[new_box] = box
                        else:
                            if level[new_man] not in ('D', 'E', 'F', 'G', 'H', 'I', 'J', 'k', 'L'):
                                if level[new_man] != str(ord(level[new_box]) - ord('A')):
                                    new_level[new_box] = trans_mix(level[new_man], level[new_box])
                            else:
                                box, _ = get_ori(level[new_man])
                                if box != str(ord(level[new_box]) - ord('A')):
                                    new_level[new_box] = trans_mix(box, level[new_box])

                        queue.append((new_level, new_man, path + direction.upper()))  # 将新状态加入队列

        return None  # 如果搜索失败，则返回空值


    def done_check(self, level):
        target_characters = {'0', '1', '2', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'}
        for char in level:
            if char in target_characters:
                return False
        return True


def choose_mode(screen, clock):
    font = pygame.font.Font(None, 36)
    modes = ["Easy", "Medium", "Hard", "One to one"]
    mode_selected = None

    while True:
        screen.fill((255, 255, 255))
        for i, mode in enumerate(modes):
            text = font.render(mode, True, (0, 0, 0))
            text_rect = text.get_rect(center=(200, 50 + i * 50))
            screen.blit(text, text_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for i, mode in enumerate(modes):
                    text_rect = font.render(mode, True, (0, 0, 0)).get_rect(center=(200, 50 + i * 50))
                    if text_rect.collidepoint(mouse_x, mouse_y):
                        mode_selected = i + 1
                        break

        if mode_selected:
            return mode_selected - 1

        clock.tick(60)


def main():
    # start pygame
    pygame.init()
    screen = pygame.display.set_mode((390, 250))
    #
    clock = pygame.time.Clock()
    pygame.key.set_repeat(200, 50)

    mode = choose_mode(screen, clock)

    # load skin
    skinfilename = os.path.join('skin.png')
    try:
        skin = pygame.image.load(skinfilename)
    except pygame.error as msg:
        print('cannot load skin')
        raise SystemExit(msg)
    skin = skin.convert()

    screen.fill(skin.get_at((0, 0)))
    pygame.display.set_caption('BoxGame.py')

    # create BoxGame object
    boxer = BoxGame(mode) if mode < 3 else BoxGame121(mode)
    boxer.draw(screen, skin)

    # path = boxer.astar_search() if mode < 3 else boxer.bfs_search_one2one()
    # print(path)
    path = path_list[mode]

    for mv in path:
        clock.tick(5)

        boxer.move(mv)
        boxer.draw(screen, skin)

        pygame.display.update()

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

        pygame.display.update()
        pygame.display.set_caption(boxer.solution.__len__().__str__() + '/' + boxer.push.__str__() + ' - boxer.py')


if __name__ == '__main__':
    main()
