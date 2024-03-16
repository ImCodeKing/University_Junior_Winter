# 导入队列库（搜索算法可能需要）
import queue
# 导入随机数库
import random


class ChessBoard:
    def __init__(self, size=8):
        '''
        初始化棋盘
        :param size: 正方形棋盘的边长尺寸，默认为8
        '''
        self.size = size
        self.board = [[' ' for _ in range(size)] for _ in range(size)]
        self.start = None
        self.end = None
        self.obstacles = []

    def set_start(self, x, y):
        '''
        设置起始点
        :param x: 起始点x坐标
        :param y: 起始点y坐标
        :return: None
        '''
        self.start = (x, y)
        self.board[x][y] = 'S'

    def set_end(self, x, y):
        '''
        设置终止点
        :param x: 终止点x坐标
        :param y: 终止点y坐标
        :return: None
        '''
        self.end = (x, y)
        self.board[x][y] = 'E'

    def add_obstacle(self, x, y):
        '''
        添加障碍点
        :param x: 障碍点x坐标
        :param y: 障碍点y坐标
        :return: None
        '''
        self.obstacles.append((x, y))

    def is_valid_move(self, x, y, prev_x, prev_y):
        '''
        判断当前移动是否有效
        :param x: 移动后点的x坐标
        :param y: 移动后点的y坐标
        :param prev_x: 移动前点的x坐标
        :param prev_y: 移动前点的y坐标
        :return: True或False
        '''
        # 移动点超出棋盘范围
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        # 已有己方棋子的地方不能下棋（B方）
        if self.board[x][y] == 'B':
            return False
        # 不允许不动
        if (x, y) == (prev_x, prev_y):
            return False

        # 当马跨越对角线的同侧长边相邻位置，检查是否有己方或敌方棋子
        diff_x = abs(x - prev_x)
        diff_y = abs(y - prev_y)
        if diff_x == 2 and diff_y == 1:
            if x - prev_x == 2:
                if self.board[prev_x + 1][prev_y] != ' ':
                    return False
            else:
                if self.board[prev_x - 1][prev_y] != ' ':
                    return False
        elif diff_x == 1 and diff_y == 2:
            if y - prev_y == 2:
                if self.board[prev_x][prev_y + 1] != ' ':
                    return False
            else:
                if self.board[prev_x][prev_y - 1] != ' ':
                    return False

        return True

    def is_goal_reached(self, x, y):
        '''
        判断当前是否抵达终点
        :param x: 当前位置的x坐标
        :param y: 当前位置的x坐标
        :return: True或False
        '''
        return (x, y) == self.end

    def get_possible_moves(self, x, y):
        '''
        获取可能的下一步的坐标
        :param x: 当前点的x坐标
        :param y: 当前点的y坐标
        :return: 所有能移动到的有效点坐标的列表
        '''
        possible_moves = [
            (x + 1, y + 2), (x + 2, y + 1),
            (x + 2, y - 1), (x + 1, y - 2),
            (x - 1, y - 2), (x - 2, y - 1),
            (x - 2, y + 1), (x - 1, y + 2)
        ]
        return [(nx, ny) for nx, ny in possible_moves if self.is_valid_move(nx, ny, x, y)]

    def print_board(self):
        '''
        打印当前的棋盘
        :return: None
        '''
        print("+" + "---+" * self.size)
        for row in self.board:
            print("|", end="")
            for cell in row:
                print(f" {cell} |", end="")
            print("\n+" + "---+" * self.size)

    def generate_random_obstacles(self, obstacles_ratio=0.3, seed=0):
        '''
        在棋盘上随机生成一定比例的障碍
        :param obstacles_ratio: 障碍棋子占棋盘位置总数的比例
        :param seed: 随机数种子，控制生成相同的障碍位置
        :return: None
        '''
        num_obstacles = int(obstacles_ratio * self.size * self.size)
        random.seed(seed)
        available_positions = [(x, y) for x in range(self.size) for y in range(self.size)]

        if self.start:
            available_positions.remove(self.start)
        if self.end:
            available_positions.remove(self.end)

        if num_obstacles > len(available_positions):
            print("生成的障碍物数量大于可用位置数量。")
            return

        self.obstacles = random.sample(available_positions, num_obstacles)
        
        for x, y in self.obstacles:
            prob = random.random()
            if prob < 0.5:
                # 红方
                self.board[x][y] = 'R'
            else:
                # 黑方
                self.board[x][y] = 'B'

    def solve(self):
        '''
        使用广度优先搜索算法解决棋盘问题
        :return: None
        '''
        if not self.start or not self.end:
            print("起始点或终止点未设置")
            return

        self.search_time = 0
        visited = set()
        q = queue.Queue()
        q.put((self.start, []))

        while not q.empty():
            (x, y), path = q.get()
            if self.is_goal_reached(x, y):
                print("跳跃步数：", len(path))
                print("跳跃路径：", path + [(x, y)])
                print("查找次数：", self.search_time)
                return

            if (x, y) in visited:
                continue
            visited.add((x, y))
            self.search_time += 1

            for next_x, next_y in self.get_possible_moves(x, y):
                if (next_x, next_y) in visited:
                    continue
                q.put(((next_x, next_y), path + [(x, y)]))

        print("未找到路径，目标不可达")


# 示例用法
if __name__ == "__main__":
    size = 100
    # 实例化一个大小为size的棋盘
    chessboard = ChessBoard(size=size)
    # 设置初始点为(0,0)
    chessboard.set_start(0, 0)
    # 设置终止点为(n-1,n-1)
    chessboard.set_end(size-1, size-1)
    # 随机生成20%的障碍棋子
    chessboard.generate_random_obstacles(obstacles_ratio=0.3, seed=4)
    # 打印初始棋盘
    print("初始棋盘：")
    chessboard.print_board()
    # 搜索算法求解
    chessboard.solve()
