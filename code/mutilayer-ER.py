import networkx as nx
import random

# 设置节点数
num_nodes = 100
# 生成第一层无标度无向图（Barabási–Albert模型）
# G1 = nx.barabasi_albert_graph(num_nodes, 3)  # 参数3表示每个新加入的节点与已有节点的连接数
# # 生成第二层无标度无向图（Barabási–Albert模型）
# G2 = nx.barabasi_albert_graph(num_nodes, 1)  # 同样使用3作为连接数
# G3 = nx.barabasi_albert_graph(num_nodes, 1)  # 同样使用3作为连接数

# 生成第一层随机无向图
G1 = nx.erdos_renyi_graph(num_nodes, 0.1)  # 0.05表示连接概率，可以根据需要调整
# 生成第二层随机无向图
G2 = nx.erdos_renyi_graph(num_nodes, 0.05)
G3 = nx.erdos_renyi_graph(num_nodes, 0.01)


# 将边存储到txt文件中，格式为 "节点1 节点2 权重"（权重设为1）
def save_network_to_txt(G, file_name):
    with open(file_name, 'w') as f:
        for edge in G.edges():
            f.write(f"{edge[0]+1} {edge[1]+1} 1\n")  # 节点编号从1开始

# 保存两层网络
save_network_to_txt(G1,  'mutilayer1.txt')
save_network_to_txt(G2,  'mutilayer2.txt')
save_network_to_txt(G3,  'mutilayer3.txt')

print("两层网络已保存到txt文件中。")
