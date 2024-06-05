import csv

# todo:指定输出文件的路径
output_file_path = 'val.txt'

# todo:打开CSV文件进行读取
with open('val_data.csv', 'r') as csvfile:
    # 创建CSV阅读器
    reader = csv.reader(csvfile)
    # 打开输出文件进行写入
    with open(output_file_path, 'w') as outfile:
        # 遍历CSV中的每一行
        for row in reader:
            # 检查行是否有足够的元素
            if len(row) >= 2:
                # todo:将图片路径和类别号写入到输出文件
                outfile.write(f"E:/project/imgs/{row[0]} {row[1]}\n")