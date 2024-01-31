def extract_text(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        lines = f_in.readlines()
        num_lines = len(lines)
        for i in range(0, num_lines, 16):
            start_index = i
            end_index = min(i + 4, num_lines)
            group_lines = lines[start_index:end_index]
            f_out.writelines(group_lines)
            #f_out.write('')  # 添加空行分隔每个组

# 指定输入文件路径和输出文件路径
input_file = '/home/air/multinerf/dataset/distributed_nerf/NewYork/1/distributed_1.txt'
output_file = '/home/air/multinerf/dataset/distributed_nerf/NewYork/1/output.txt'

# 调用函数提取文本
extract_text(input_file, output_file)