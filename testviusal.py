def visualize_guitar_tabs(tab_sequence):
    strings = ["E", "A", "D", "G", "B", "E"]  # 标准吉他弦，从低音到高音
    num_measures = len(tab_sequence) // 4  # 每四个 tuple 组成一个小节
    
    # 初始化 TAB 结构
    tab_lines = {string: "" for string in strings}  # 高音 E 在最上
    finger_lines = {string: "" for string in strings}  # 手指分配行
    
    for measure in range(num_measures):
        for i in range(4):  # 每个小节 4 个和弦
            fret_press, finger_assign = tab_sequence[measure * 4 + i]
            for j, (fret, finger) in enumerate(zip(fret_press, finger_assign)):
                fret_str = "-" if fret == -1 else str(fret)
                finger_str = "-" if finger == -1 else str(finger)
                
                # 确保所有字符宽度一致
                tab_lines[strings[j]] += fret_str.ljust(4, '-')
                finger_lines[strings[j]] += finger_str.ljust(4, '-')
        
        # 添加小节线
        for string in strings:
            tab_lines[string] += "|"
            finger_lines[string] += "|"
    
    # 计算每根弦的最大长度
    max_length = max(len(line) for line in tab_lines.values())
    
    # 填充每根弦，使其长度一致
    for string in strings:
        tab_lines[string] = tab_lines[string].ljust(max_length, '-')
        finger_lines[string] = finger_lines[string].ljust(max_length, '-')
    
    # 打印手指分配
    print("Fingering:")
    for string in strings[::-1]:  # 高音在上
        print(f"{string}|{finger_lines[string]}")
    print("\nTAB:")
    for string in strings[::-1]:
        print(f"{string}|{tab_lines[string]}")

# 示例输入
tab_sequence = [
    ([0, 6, -1, 0, -1, 0], [-1, 3, -1, -1, -1, -1]),
    ([-1, 3, -1, -1, 7, 0], [-1, 3, -1, -1, 1, -1]),
    ([0, -1, -1, -1, -1, 0], [-1, -1, -1, -1, -1, -1]),
    ([-1, 5, 2, -1, 0, -1], [-1, 2, 4, -1, -1, -1]),
    ([-1, -1, 0, -1, -1, 0], [-1, -1, -1, -1, -1, -1]),
    ([0, 0, 0, 8, 0, -1], [-1, -1, -1, 3, -1, -1]),
    ([0, -1, 4, 0, -1, -1], [-1, -1, 4, -1, -1, -1]),
    ([-1, 0, 0, 4, 0, 0], [-1, -1, -1, 4, -1, -1])
]

visualize_guitar_tabs(tab_sequence)