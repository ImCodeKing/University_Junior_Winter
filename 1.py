# If you need to import additional packages or classes, please import here.
class CELL():
    def __init__(self, time, count, neighbor=None):
        self.used_count = 0
        self.used_time = time
        self.neighbor = None


def func():
    CELL_dict = {}
    time = 0
    while True:
        time += 1
        words = input().split()
        if words[0] == "capacity:":
            capacity = int(input())
        elif words[0] == "write:":
            write_count = int(input())
            for _ in range(write_count):
                if len(CELL_dict) + write_count > capacity:
                    min_time = 1000000000
                    min_count = 1000000000
                    for key in CELL_dict.keys():
                        item = CELL_dict[key]
                        if item.used_count <= min_count:
                            if item.used_time < min_time:
                                min_count = item.used_count
                                min_time = item.used_time
                                min_key = key
                    CELL_dict.pop(min_key)
                write = input().split()
                if write[0] not in CELL_dict.keys():
                    CELL_dict[write[0]] = CELL(time, 1)
                else:
                    CELL_dict[write[0]].used_count += 1
                    CELL_dict[write[0]].used_time == time
                CELL_dict[write[0]].neighbor = write[1]

                #if CELL_dict[write[0]].neighbor in CELL_dict.keys():
                #    if CELL_dict[CELL_dict[write[0]].neighbor] is not None:
                #        CELL_dict[CELL_dict[write[0]].neighbor].used_count += 1
                #        CELL_dict[CELL_dict[write[0]].neighbor].used_time == time
        elif words[0] == "read:":
            read = input()
            CELL_dict[read].used_count += 1
            CELL_dict[read].used_time == time
            if CELL_dict[read].neighbor in CELL_dict.keys():
                if CELL_dict[CELL_dict[read].neighbor] is not None:
                    CELL_dict[CELL_dict[read].neighbor].used_count += 1
                    CELL_dict[CELL_dict[read].neighbor].used_time == time
        elif words[0] == "query:":
            query = input()
            if query in CELL_dict.keys():
                print(CELL_dict[query].neighbor)
            else:
                print(-1)


if __name__ == '__main__':
        func()
