import pygame, sys, os
from pygame.locals import *

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


class BoxGame:
    def __init__(self):
        self.level = list(
            '----#####--------------#---#--------------#$--#------------###--$##-----------#--$-$-#---------###-#-##-#---#######---#-##-#####--..##-$--$----------..######-###-#@##--..#----#-----#########----#######--------')
        self.w = 19
        self.h = 11
        self.man = 163
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
        elif self.level[self.man + h] == '*' or self.level[self.man + h] == '$':
            if self.level[self.man + h2] == '-' or self.level[self.man + h2] == '.':
                # push
                to_box(self.level, self.man + h2)
                to_man(self.level, self.man + h)
                to_floor(self.level, self.man)
                self.man += h
                self.solution += d.upper()
                self.push += 1

    def automove(self):
        if self.auto == 1 and self.todo.__len__() > 0:
            self._move(self.todo[-1].lower())
            self.todo.pop()
        else:
            self.auto = 0

    def boxhint(self, x, y):
        d4 = [-1, -self.w, 1, self.w]
        m4 = ['l', 'u', 'r', 'd']
        b = y * self.w + x
        maze = list(self.level)
        to_floor(maze, b)
        to_floor(maze, self.man)
        mark = maze * 4
        size = self.w * self.h
        self.queue = []
        head = 0
        for i in range(4):
            if b_manto(maze, self.w, b, self.man, b + d4[i]):
                if len(self.queue) == 0:
                    self.queue.append((b, i, -1))
                mark[i * size + b] = '1'
        # print self.queue
        while head < len(self.queue):
            pos = self.queue[head]
            head += 1
            # print pos
            for i in range(4):
                if mark[pos[0] + i * size] == '1' and maze[pos[0] - d4[i]] in ['-', '.']:
                    # print i
                    if mark[pos[0] - d4[i] + i * size] != '1':
                        self.queue.append((pos[0] - d4[i], i, head - 1))
                        for j in range(4):
                            if b_manto(maze, self.w, pos[0] - d4[i], pos[0], pos[0] - d4[i] + d4[j]):
                                mark[j * size + pos[0] - d4[i]] = '1'
        for i in range(size):
            self.hint[i] = '0'
            for j in range(4):
                if mark[j * size + i] == '1':
                    self.hint[i] = '1'
        # print self.hint


import heapq

# 定义游戏状态类
class State:
    def __init__(self, level, player_pos, box_pos):
        self.level = level
        self.player_pos = player_pos
        self.box_pos = box_pos
        self.g = 0  # 从起始状态到当前状态的实际代价
        self.h = 0  # 启发式评估函数的值
        self.parent = None  # 父状态，用于回溯路径

    # 定义状态的比较方法，用于优先队列的排序
    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

# 定义启发式评估函数，这里使用曼哈顿距离
def heuristic(state, goal_pos):
    total_distance = 0
    for box, goal in zip(state.box_pos, goal_pos):
        total_distance += abs(box[0] - goal[0]) + abs(box[1] - goal[1])
    return total_distance

# 将地图状态转换为游戏状态
def map_to_game_state(level):
    player_pos = None
    box_pos = []
    goal_pos = []
    for j in range(len(level)):
        for i in range(len(level[j])):
            if level[j][i] == '@':
                player_pos = (i, j)
            elif level[j][i] == '$':
                box_pos.append((i, j))
            elif level[j][i] == '.':
                goal_pos.append((i, j))
    return player_pos, box_pos, goal_pos

# 定义A*搜索函数
def astar_search(level, start_player_pos, start_box_pos, goal_pos):
    open_list = []  # 优先队列，存放待探索的状态
    closed_set = set()  # 存放已探索过的状态

    start_state = State(level, start_player_pos, start_box_pos)
    heapq.heappush(open_list, start_state)

    while open_list:
        current_state = heapq.heappop(open_list)

        if current_state.box_pos == goal_pos:
            # 找到了目标状态，回溯路径并返回
            path = []
            while current_state:
                path.append(current_state)
                current_state = current_state.parent
            return path[::-1]

        closed_set.add(current_state)

        # 展开当前状态的相邻状态
        for move in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            new_box_pos = []
            for box in current_state.box_pos:
                new_player_pos = (box[0] + move[0], box[1] + move[1])
                new_box_pos.append(new_player_pos)

            if any(new_box == current_state.player_pos for new_box in new_box_pos):
                # 玩家试图推箱子到当前箱子的位置，不合法的移动
                continue

            new_level = [list(row) for row in current_state.level]  # 复制地图状态
            for i, box in enumerate(current_state.box_pos):
                new_level[box[1]][box[0]] = '-'  # 清除原箱子位置
                new_level[new_box_pos[i][1]][new_box_pos[i][0]] = '$'  # 更新新箱子位置

            new_state = State(new_level, current_state.player_pos, new_box_pos)
            new_state.g = current_state.g + 1
            new_state.h = heuristic(new_state, goal_pos)
            new_state.parent = current_state

            if new_state not in closed_set:
                heapq.heappush(open_list, new_state)

    # 如果搜索失败，返回空路径
    return []

def search(level):
    # 示例用法
    # level = [
    #     '----#####--------------#---#--------------#$--#------------###--$##-----------#--$-$-#---------###-#-##-#---#######---#-##-#####--..##-$--$----------..######-###-#@##--..#----#-----#########----#######--------'
    # ]

    start_player_pos, start_box_pos, goal_pos = map_to_game_state(level)
    path = astar_search(level, start_player_pos, start_box_pos, goal_pos)
    if path:
        for state in path:
            print("Player Position:", state.player_pos, "Box Positions:", state.box_pos)
    else:
        print("No solution found.")




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

    # print skin.get_at((0,0))
    # screen.fill((255,255,255))
    screen.fill(skin.get_at((0, 0)))
    pygame.display.set_caption('BoxGame.py')

    # create BoxGame object
    boxer = BoxGame()
    boxer.draw(screen, skin)

    #
    clock = pygame.time.Clock()
    pygame.key.set_repeat(200, 50)

    # main game loop
    while True:
        search(boxer.level)
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
        pygame.display.set_caption(boxer.solution.__len__().__str__() + '/' + boxer.push.__str__() + ' - sokoban.py')


if __name__ == '__main__':
    main()
