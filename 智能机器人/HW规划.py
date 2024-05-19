import numpy as np
import matplotlib.pyplot as plt

# 定义场景
start = (1, 1)
goal = (9, 9)
obstacles = [(3, 3), (3, 4), (5, 5), (6, 5), (7, 7), (8, 8), (7, 8), (8, 7)]


# 画出场景
def plot_scene(path=None):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # 画出障碍物
    for obs in obstacles:
        ax.add_patch(plt.Rectangle(obs, 1, 1, color='black'))

    # 画出起点和终点
    ax.add_patch(plt.Circle(start, 0.1, color='yellow'))
    ax.add_patch(plt.Circle(goal, 0.1, color='red'))

    # 画出路径
    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color='blue')

    plt.grid(True)
    plt.show()


plot_scene()


def is_obstacle(x, y):
    for obs in obstacles:
        if obs[0] <= x < obs[0] + 1 and obs[1] <= y < obs[1] + 1:
            return True
    return False


def get_neighbors(x, y):
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]


def bug1_algorithm(start, goal):
    current_pos = start
    path = [current_pos]
    hit_point = None

    while current_pos != goal:
        while current_pos != goal:
            new_pos = (current_pos[0] + np.sign(goal[0] - current_pos[0]),
                       current_pos[1] + np.sign(goal[1] - current_pos[1]))
            if is_obstacle(new_pos[0], new_pos[1]):
                hit_point = current_pos
                break
            current_pos = new_pos
            path.append(current_pos)

        if current_pos == goal:
            break

        min_dist_to_goal = float('inf')
        best_pos = None

        # 绕行障碍物
        while True:
            for neighbor in get_neighbors(current_pos[0], current_pos[1]):
                if not is_obstacle(neighbor[0], neighbor[1]) and neighbor != path[-1]:
                    dist_to_goal = np.linalg.norm(np.array(neighbor) - np.array(goal))
                    if dist_to_goal < min_dist_to_goal:
                        min_dist_to_goal = dist_to_goal
                        best_pos = neighbor

            if best_pos is None or best_pos == hit_point:
                break

            current_pos = best_pos
            path.append(current_pos)

        if best_pos:
            current_pos = best_pos

    return path


def bug2_algorithm(start, goal):
    current_pos = start
    path = [current_pos]
    direction_to_goal = (goal[0] - start[0], goal[1] - start[1])
    m_line_slope = direction_to_goal[1] / direction_to_goal[0]

    while current_pos != goal:
        # 直接朝目标方向移动
        while current_pos[0] != goal[0] or current_pos[1] != goal[1]:
            new_pos = (current_pos[0] + np.sign(goal[0] - current_pos[0]),
                       current_pos[1] + np.sign(goal[1] - current_pos[1]))
            if is_obstacle(new_pos[0], new_pos[1]):
                break
            current_pos = new_pos
            path.append(current_pos)

        # 如果已经到达目标点，结束
        if current_pos == goal:
            break

        # 沿障碍物边界绕行直到再次碰到m-line
        boundary_pos = current_pos
        while True:
            new_pos = (boundary_pos[0] + np.sign(goal[0] - boundary_pos[0]),
                       boundary_pos[1] + np.sign(goal[1] - boundary_pos[1]))
            if not is_obstacle(new_pos[0], new_pos[1]):
                boundary_pos = new_pos
                path.append(boundary_pos)
                if (boundary_pos[1] - start[1]) == m_line_slope * (boundary_pos[0] - start[0]):
                    break
                if boundary_pos == current_pos:
                    break

        current_pos = boundary_pos

    return path


if __name__ == '__main__':
    path_bug1 = bug1_algorithm(start, goal)
    plot_scene(path_bug1)


    # path_bug2 = bug2_algorithm(start, goal)
    # plot_scene(path_bug2)
