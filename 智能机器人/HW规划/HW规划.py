import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import colors
import queue

# 起点和终点
start = (1, 1)
goal = (9, 9)

# 添加障碍物
obstacles = [(3, 3), (4, 3), (5, 5), (5, 6), (7, 7), (8, 8), (7, 8), (8, 7)]

grid_size = 10


def Move(current):
    dx = goal[0] - current[0]
    dy = goal[1] - current[1]

    if abs(dx) >= abs(dy):
        if dx > 0:
            d = [1, 0]
        else:
            d = [-1, 0]
    else:
        if dy > 0:
            d = [0, 1]
        else:
            d = [0, -1]

    next_x = current[0] + d[0]
    next_y = current[1] + d[1]
    return next_x, next_y


def get_possible_moves(x, y, available):
    possible_moves = [
        (x + 1, y), (x - 1, y),
        (x, y + 1), (x, y - 1),
    ]
    return [(nx, ny) for nx, ny in possible_moves if (nx, ny) not in obstacles and (nx, ny) in available]


def BFS(start, goal, available):
    reached = []

    q = queue.Queue()
    # 加入起点
    q.put((start, [start]))

    while not q.empty():
        # Get the current position and path
        current_position, path = q.get()

        if current_position in reached:
            continue
        reached.append(current_position)

        # 如果当前节点是终点，返回路径
        if current_position == goal:
            return path

        # 当搜索到终点时，回溯得到路径
        for move in get_possible_moves(*current_position, available):
            # 如果不是已经到过的点，加入队列
            if move not in reached:
                q.put((move, path + [move]))
    return None


def visual(current=start, path=None):
    fig, ax = plt.subplots()
    board_visual = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            if path is not None and (i, j) in path:
                board_visual[i - 1][j - 1] = 1
            elif (i, j) == start:
                board_visual[i - 1][j - 1] = 1
            elif (i, j) in obstacles:
                board_visual[i - 1][j - 1] = 4
            if (i, j) == goal:
                board_visual[i][j] = 3
    if current:
        board_visual[current[0] - 1][current[1] - 1] = 2
        ax.plot(current[1], current[0], 'yo', markersize=10)

    ax.plot(start[1], start[0], 'yo', markersize=10, )
    ax.plot(goal[1], goal[0], 'ro', markersize=10)

    if path:
        for p in path:
            ax.plot(p[1], p[0], 'go', markersize=10)

    cmap = colors.ListedColormap(['white', 'red', 'yellow', 'purple', 'black'])

    ax.set_xticks(np.arange(0, board_visual.shape[1], 1))
    ax.set_yticks(np.arange(0, board_visual.shape[0], 1))

    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))

    ax.set_ylim(bottom=grid_size)
    ax.set_ylim(top=0.)
    ax.set_xlim(left=0.)
    ax.set_xlim(right=grid_size)

    ax.imshow(board_visual, cmap=cmap, extent=[0, 10, 10, 0], aspect='auto')
    plt.grid(True)
    plt.show()


def get_surrd_point(obstacles_discover, radius=1, current=None):
    surrd_point = []
    min_cost_point = None
    min_cost = float('inf')
    for obst in obstacles_discover:
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                new_x = obst[0] + i
                new_y = obst[1] + j
                if (new_x, new_y) not in obstacles:
                    surrd_point.append((new_x, new_y))
                    # Calculate cost for each surrd_point point
                    if current is not None:
                        cost = abs(new_x - current[0]) + abs(new_y - current[1])
                    else:
                        cost = abs(new_x - goal[0]) + abs(new_y - goal[1])
                    if cost < min_cost:
                        min_cost = cost
                        min_cost_point = (new_x, new_y)
    if current is not None:
        return surrd_point, min_cost_point
    else:
        return list(set(surrd_point)), min_cost_point


def discover(current):
    obstacles_discover = []
    for i in [1, -1]:
        new_x = current[0] + i
        new_y = current[1]
        if (new_x, new_y) in obstacles:
            obstacles_discover.append((new_x, new_y))
    for j in [1, -1]:
        new_x = current[0]
        new_y = current[1] + j
        if (new_x, new_y) in obstacles:
            obstacles_discover.append((new_x, new_y))
    return obstacles_discover


def obstacleConnect(new_obstacles, obstacles_discover):
    for obst in new_obstacles:
        for point in obstacles_discover:
            if abs(obst[0] - point[0]) + abs(obst[1] - point[1]) == 1:
                return True
    return False


def bug(start, goal):
    current = start
    path = [current]
    while current != goal:
        Next = Move(current)
        if Next not in obstacles:
            current = Next
            path.append(current)

        else:
            visual(current, path)
            tmp = current
            obstacles_discover = [Next]
            reached = [current]

            surrd_point, min_cost_point = get_surrd_point(obstacles_discover)

            while set(reached) != set(surrd_point):
                nearest = np.inf
                for point in surrd_point:
                    if point not in reached:
                        distance = abs(current[0] - point[0]) + abs(current[1] - point[1])
                        if distance < nearest:
                            nearest = distance
                            nearest_point = point
                if (BFS(current, nearest_point, surrd_point)):
                    current = nearest_point
                    reached.append(current)
                    new_obstacles = discover(current)
                    if new_obstacles != []:
                        if obstacleConnect(new_obstacles, obstacles_discover):
                            obstacles_discover.extend(new_obstacles)
                            obstacles_discover = list(set(obstacles_discover))
                    surrd_point, min_cost_point = get_surrd_point(obstacles_discover)
            print(BFS(tmp, min_cost_point, surrd_point))
            path.extend(BFS(tmp, min_cost_point, surrd_point))
            current = min_cost_point

            visual(current, path)
    return path


def bug2(start, goal):
    current = start
    path = [current]
    correct_d = (np.array(start) - np.array(goal)) / np.linalg.norm(np.array(start) - np.array(goal))

    while current != goal:
        Next = Move(current)
        if Next not in obstacles:
            current = Next
            path.append(current)
        else:
            visual(current, path)
            tmp = current
            obstacles_discover = [Next]
            reached = [current]
            surrd_point, min_cost_point = get_surrd_point(obstacles_discover)
            while set(reached) != set(surrd_point):
                nearest = np.inf
                for point in surrd_point:
                    if point not in reached:
                        distance = abs(current[0] - point[0]) + abs(current[1] - point[1])
                        if distance == 1:
                            nearest = distance
                            nearest_point = point
                            break
                        if distance < nearest:
                            nearest = distance
                            nearest_point = point

                if (BFS(current, nearest_point, surrd_point)):
                    current = nearest_point
                    reached.append(current)
                    new_obstacles = discover(current)
                    if new_obstacles != []:
                        if obstacleConnect(new_obstacles, obstacles_discover):
                            obstacles_discover.extend(new_obstacles)
                            surrd_point, _ = get_surrd_point(obstacles_discover)
                if np.cross(correct_d, (np.array(current) - np.array(goal)) / np.linalg.norm(np.array(current) - np.array(goal)) if current == goal else [0, 0]) == 0 and current not in path:
                    path.extend(BFS(tmp, current, surrd_point))
                    visual(current, path)
                    break

    return path


visual()
# 运行BUG1算法
# path_bug = bug(start, goal)
# print("BUG Path:", path_bug)
# visual(current=None, path=path_bug)

# 运行BUG2算法
path_bug2 = bug2(start, goal)
print("BUG2 Path:", path_bug2)
visual(current=None, path=path_bug2)
