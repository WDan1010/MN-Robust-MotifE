
import networkx as nx
import matplotlib.pyplot as plt


def read_network(file_path):
    """从文件中读取网络数据，构建图结构"""
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            node1, node2, weight = map(int, line.strip().split())
            G.add_edge(node1, node2, weight=weight)
    return G


def intentional_attack(G, attack_fraction):
    """按节点度值大小攻击网络G中的节点，按比例attack_fraction移除节点及其相关边"""
    total_nodes = list(G.nodes())
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    num_nodes_to_attack = int(len(total_nodes) * attack_fraction)
    nodes_to_attack = [node for node, _ in sorted_nodes[:num_nodes_to_attack]]
    G.remove_nodes_from(nodes_to_attack)
    return set(nodes_to_attack)



def intentional_attack(G, attack_fraction):
    """按节点度值大小攻击网络G中的节点，按比例attack_fraction移除节点及其相关边"""
    total_nodes = list(G.nodes())
    # 按节点度值从大到小排序
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
    num_nodes_to_attack = int(len(total_nodes) * attack_fraction)
    # 选择度值最大的节点进行攻击
    nodes_to_attack = [node for node, _ in sorted_nodes[:num_nodes_to_attack]]
    G.remove_nodes_from(nodes_to_attack)
    return set(nodes_to_attack)



def remove_node_and_edges(G, node):
    """移除节点及其连接的边，并返回与该节点相连的所有邻居节点。"""
    neighbors = list(G.neighbors(node))
    G.remove_node(node)
    return neighbors

def check_isolated_nodes(G, neighbors, failed_nodes):
    """检查邻居节点是否变为孤立，并返回新的失效节点。"""
    new_failed_nodes = set()
    for neighbor in neighbors:
        if neighbor not in failed_nodes and G.degree(neighbor) == 0:
            new_failed_nodes.add(neighbor)
    return new_failed_nodes

def propagate_failure_between_layers(source_layer, target_layer, G_source, G_target, coupling_strength, threshold, failed_nodes):
    """
    根据耦合强度和阈值在网络之间双向传播故障。
    """
    new_failed_nodes = set()
    if coupling_strength > threshold:
        for node in failed_nodes:
            if node in G_target.nodes():
                # 移除节点及其连接的边
                neighbors = remove_node_and_edges(G_target, node)
                new_failed_nodes.add(node)
                # 检查邻居节点是否变为孤立
                new_failed_nodes.update(check_isolated_nodes(G_target, neighbors, failed_nodes))
        if new_failed_nodes:
            print(f"  从第{source_layer}层传播到第{target_layer}层，新失效节点：{new_failed_nodes}")
        else:
            print(f"  从第{source_layer}层传播到第{target_layer}层，无新失效节点")
    else:
        print(f"  耦合强度 {coupling_strength} <= 阈值 {threshold}，未传播故障到第{target_layer}层")
    return new_failed_nodes

def propagate_failure_3layers(G1, G2, G3, couplings, thresholds, failed_nodes, attack_layer):
    """根据耦合强度和阈值在五层网络之间双向传播故障。"""
    failed_nodes_per_layer = {1: set(), 2: set(), 3: set()}
    failed_nodes_per_layer[attack_layer] = set(failed_nodes)

    # 定义层的耦合关系
    coupling_matrix = {
        (1, 2): couplings['12'],
        (2, 1): couplings['21'],
        (1, 3): couplings['13'],
        (3, 1): couplings['31'],
        (2, 3): couplings['23'],
        (3, 2): couplings['32']

    }

    # 存储是否有新的节点失效（用于级联判断）
    new_failures = True

    # 按顺序传播
    while new_failures:
        new_failures = False  # 初始化为False，直到检测到新失效节点

        # 检查每层的失效节点是否会影响其他层
        for source_layer, target_layer in coupling_matrix:
            if failed_nodes_per_layer[source_layer]:  # 如果源层有失效节点
                new_failed_nodes = propagate_failure_between_layers(
                    source_layer, target_layer,
                    eval(f'G{source_layer}'), eval(f'G{target_layer}'),
                    coupling_matrix[(source_layer, target_layer)],
                    thresholds[f'{source_layer}{target_layer}'],
                    failed_nodes_per_layer[source_layer]
                )
                if new_failed_nodes:
                    failed_nodes_per_layer[target_layer].update(new_failed_nodes)
                    new_failures = True

        # 反向传播：检查目标层的新失效节点是否影响源层
        for target_layer, source_layer in coupling_matrix.items():
            if failed_nodes_per_layer[target_layer[1]]:  # 如果目标层有失效节点
                new_failed_nodes = propagate_failure_between_layers(
                    target_layer[1], target_layer[0],
                    eval(f'G{target_layer[1]}'), eval(f'G{target_layer[0]}'),
                    coupling_matrix[(target_layer[1], target_layer[0])],
                    thresholds[f'{target_layer[1]}{target_layer[0]}'],
                    failed_nodes_per_layer[target_layer[1]]
                )
                if new_failed_nodes:
                    failed_nodes_per_layer[target_layer[0]].update(new_failed_nodes)
                    new_failures = True

    # 返回最终的失效网络及每层的失效节点
    return G1, G2, G3, failed_nodes_per_layer


def get_max_connected_component_size(G):
    """获取图G中最大连通分量的大小"""
    if len(G) == 0:
        return 0
    return len(max(nx.connected_components(G), key=len))


def save_network_to_file(G, file_path):
    """将图G的节点和边数据保存到文件中"""
    with open(file_path, 'w') as f:
        for u, v, data in G.edges(data=True):
            f.write(f"{u} {v} {data['weight']}\n")


def draw_max_connected_component(G, layer_number, output_file):
    """画出网络G中最大连通子图"""
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    plt.figure()
    nx.draw(subgraph, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
    plt.title(f"Layer {layer_number} - Largest Connected Component")
    plt.savefig(output_file)
    plt.close()


def simulate_multiplex_attack(file1, file2, file3,couplings, thresholds, attack_layer, attack_ratio, output_prefix):
    """模拟三层网络攻击，并记录结果和保存剩余网络"""
    # 读取网络
    G1 = read_network(file1)
    G2 = read_network(file2)
    G3 = read_network(file3)


    # 按度值攻击选定层
    if attack_layer == 1:
        initial_failed_nodes = intentional_attack(G1, attack_ratio)
    elif attack_layer == 2:
        initial_failed_nodes = intentional_attack(G2, attack_ratio)
    elif attack_layer == 3:
        initial_failed_nodes = intentional_attack(G3, attack_ratio)

    # 传播故障到其他层，并进行层间级联传播
    G1_final, G2_final, G3_final, failed_nodes_per_layer = propagate_failure_3layers(G1, G2, G3, couplings, thresholds, initial_failed_nodes, attack_layer)

    # 获取每层网络最大连通分量的大小
    max_connected_size_G1 = get_max_connected_component_size(G1_final)
    max_connected_size_G2 = get_max_connected_component_size(G2_final)
    max_connected_size_G3 = get_max_connected_component_size(G3_final)


    # 保存攻击后的剩余网络数据
    save_network_to_file(G1_final, f"{output_prefix}_layer1_after_attack.txt")
    save_network_to_file(G2_final, f"{output_prefix}_layer2_after_attack.txt")
    save_network_to_file(G3_final, f"{output_prefix}_layer3_after_attack.txt")


    # # 画出每层的最大连通子图
    # draw_max_connected_component(G1_final, 1, f"{output_prefix}_layer1_largest_cc.png")
    # draw_max_connected_component(G2_final, 2, f"{output_prefix}_layer2_largest_cc.png")
    # draw_max_connected_component(G3_final, 3, f"{output_prefix}_layer3_largest_cc.png")


    return {
        "Max_Connected_Size_G1": max_connected_size_G1,
        "Max_Connected_Size_G2": max_connected_size_G2,
        "Max_Connected_Size_G3": max_connected_size_G3,
        "Failed_Nodes_G1": failed_nodes_per_layer[1],
        "Failed_Nodes_G2": failed_nodes_per_layer[2],
        "Failed_Nodes_G3": failed_nodes_per_layer[3]
    }


# 设置文件路径
file1 = "mutilayer1.txt"
file2 = "mutilayer2.txt"
file3 = "mutilayer3.txt"


# 网络之间的耦合强度，固定值
couplings = {
    '12': 0.9035,  # 第一层对第二层的耦合强度
    '21': 0.8969,  # 第二层对第一层的耦合强度
    '13': 0.7155,
    '31': 0.3121,
    '23': 0.6945,
    '32': 0.3143
}

# 耦合强度阈值
thresholds = {
    '12': 0.1,
    '21': 0.1,
    '13': 0.1,
    '31': 0.1,
    '23': 0.1,
    '32': 0.1
}

# 攻击层（可以选择1, 2 或 3）
attack_layer = 2

# 攻击比例（例如10%）
attack_ratio = 0.46

# 输出文件前缀
output_prefix = "ER"

# 运行模拟
results = simulate_multiplex_attack(file1, file2, file3, couplings, thresholds, attack_layer, attack_ratio,
                                    output_prefix)

# 打印结果
print(f"Attack Layer: {attack_layer}, Attack Ratio: {attack_ratio}")
print(f"  Max Connected Size G1: {results['Max_Connected_Size_G1']}")
print(f"  Max Connected Size G2: {results['Max_Connected_Size_G2']}")
print(f"  Max Connected Size G3: {results['Max_Connected_Size_G3']}")
print(f"  Failed Nodes in Layer 1: {results['Failed_Nodes_G1']}")
print(f"  Failed Nodes in Layer 2: {results['Failed_Nodes_G2']}")
print(f"  Failed Nodes in Layer 3: {results['Failed_Nodes_G3']}")
