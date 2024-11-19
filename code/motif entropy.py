import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
from networkx.algorithms.isomorphism import GraphMatcher
import math

# 定义文件路径
file_path ='ER_layer2_after_attack.txt'

# 尝试读取文件，处理潜在的解析错误
try:
    # 尝试使用空格作为分隔符，并跳过有问题的行
    edges = pd.read_csv(file_path, delimiter=r'\s+', header=None, names=['node1', 'node2', 'weight'],
                        engine='python', on_bad_lines='skip')  # 对于 pandas >= 1.3.0
    # 对于 pandas < 1.3.0，可以使用 error_bad_lines=False
    # edges = pd.read_csv(file_path, delimiter=r'\s+', header=None, names=['node1', 'node2', 'weight'],
    #                     engine='python', error_bad_lines=False)
except pd.errors.ParserError as e:
    print("解析文件时出错：", e)
    # 进一步处理或退出
    exit()

# 打印读取的数据，检查是否正确
print("读取的边数据：")
print(edges.head())

# 创建带权图
G = nx.Graph()
for idx, row in edges.iterrows():
    try:
        G.add_edge(row['node1'], row['node2'], weight=row['weight'])
    except Exception as e:
        print(f"添加边时出错（行 {idx + 1}）：", e)

# 打印图中的节点和边
print(f"图中有 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")

# 创建一个函数来根据邻接矩阵创建无向图
def adj_matrix_to_graph(adj_matrix):
    G = nx.Graph()
    n = len(adj_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)
    return G

# 定义要查找的模体邻接矩阵
# CS-5
motifs = {
    'ER12_1': np.array([[0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]]),
    # 'ER3_1': np.array([[0, 0, 1],
    #                    [0, 0, 1],
    #                    [1, 1, 0]]),

    #ER1
    # 'ER1_2': np.array([[0, 0, 0, 1],
    #                    [0, 0, 1, 1],
    #                    [0, 1, 0, 1],
    #                    [1, 1, 1, 0]]),
    # 'ER1_3': np.array([[0, 0, 1, 1],
    #                    [0, 0, 1, 1],
    #                    [1, 1, 0, 1],
    #                    [1, 1, 1, 0]]),


    # ER2
    'ER2_2': np.array([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0]]),
    'ER2_3': np.array([[0, 0, 0, 1],
                       [0, 0, 1, 1],
                       [0, 1, 0, 1],
                       [1, 1, 1, 0]]),

}

# 打印模体邻接矩阵
for motif_name, motif_matrix in motifs.items():
    print(f"{motif_name} 模体的邻接矩阵：\n{motif_matrix}")

motif_counts = {motif_name: 0 for motif_name in motifs}
node_motif_degree = defaultdict(int)

# 遍历所有可能的三节点子图
for nodes in combinations(G.nodes(), 3):
    subgraph = G.subgraph(nodes)
    adj_matrix = nx.to_numpy_array(subgraph, nodelist=nodes)
    adj_matrix_binary = (adj_matrix > 0).astype(int)

    # 将当前子图转化为无向图
    subgraph_obj = adj_matrix_to_graph(adj_matrix_binary)

    # 检查三节点模体
    for motif_name, motif_matrix in motifs.items():
        if motif_matrix.shape == (3, 3):
            motif_graph = adj_matrix_to_graph(motif_matrix)
            GM = GraphMatcher(subgraph_obj, motif_graph)
            if GM.is_isomorphic():
                motif_counts[motif_name] += 1
                for node in subgraph.nodes():
                    node_motif_degree[node] += 1

# 遍历所有可能的四节点子图
for nodes in combinations(G.nodes(), 4):
    subgraph = G.subgraph(nodes)
    adj_matrix = nx.to_numpy_array(subgraph, nodelist=nodes)
    adj_matrix_binary = (adj_matrix > 0).astype(int)

    # 将当前子图转化为无向图
    subgraph_obj = adj_matrix_to_graph(adj_matrix_binary)

    # 检查四节点模体
    for motif_name, motif_matrix in motifs.items():
        if motif_matrix.shape == (4, 4):
            motif_graph = adj_matrix_to_graph(motif_matrix)
            GM = GraphMatcher(subgraph_obj, motif_graph)
            if GM.is_isomorphic():
                motif_counts[motif_name] += 1
                for node in subgraph.nodes():
                    node_motif_degree[node] += 1

# 输出模体出现次数
print("\n模体出现次数：")
for motif_name, count in motif_counts.items():
    print(f"{motif_name} 模体出现了 {count} 次")

# 输出每个节点的模体顶点度
print("\n每个节点参与的模体次数（模体顶点度）：")
for node, degree in node_motif_degree.items():
    print(f"节点 {node} 参与了 {degree} 个模体")

# 计算网络中每个节点的度
node_degrees = dict(G.degree(weight=None))  # 节点度（无权重）

# 将模体顶点度和节点度整理为DataFrame
data = pd.DataFrame({
    'node': list(G.nodes()),
    'degree': [node_degrees.get(node, 0) for node in G.nodes()],
    'motif_degree': [node_motif_degree.get(node, 0) for node in G.nodes()]
})

# 打印节点度和模体度数据
print(data)

# 进行相关性检验
pearson_corr, pearson_p = pearsonr(data['degree'], data['motif_degree'])
spearman_corr, spearman_p = spearmanr(data['degree'], data['motif_degree'])

# 输出相关性结果
print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p}")
print(f"Spearman correlation: {spearman_corr}, p-value: {spearman_p}")
# 计算模体熵
total_motif_degree = sum(node_motif_degree.values())  # 总模体度
if total_motif_degree > 0:
    # 计算每个节点的模体度概率
    data['motif_degree_prob'] = data['motif_degree'] / total_motif_degree
    # 计算熵
    data['entropy_component'] = -data['motif_degree_prob'] * np.log2(data['motif_degree_prob'])
    # 总熵（忽略概率为零的节点）
    motif_entropy = data.loc[data['motif_degree_prob'] > 0, 'entropy_component'].sum()
else:
    motif_entropy = 0


# 打印模体熵结果
print(f"\n模体熵（Shannon Entropy）：{motif_entropy:.4f}")

