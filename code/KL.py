import numpy as np
import pandas as pd
import networkx as nx
import math
from scipy.stats import entropy, pearsonr

#
# import csv
#
# # 定义一个函数将 CSV 数据转换为 TXT 数据
# def csv_to_txt(input_file, output_file):
#     with open(input_file, 'r') as csvfile, open(output_file, 'w') as txtfile:
#         reader = csv.reader(csvfile)
#         # 跳过表头（如果 CSV 文件没有表头，请将此行注释掉）
#         next(reader)
#         for row in reader:
#             # 将每一行数据格式化为 “节点1 节点2 权重” 的格式
#             txtfile.write(f"{row[0]} {row[1]} {row[2]}\n")
#
# # 假设我们导入了一个 CSV 文件到 'data.csv'
# csv_file_path = 'mutilayer3.csv'  # 上传的 CSV 文件路径
# txt_file_path = 'mutilayer3.txt'  # 保存的 TXT 文件路径
#
# # 调用函数将 CSV 文件转换为 TXT 文件
# csv_to_txt(csv_file_path, txt_file_path)
#
# # 提供生成的 TXT 文件供下载
# txt_file_path





def load_network(file_path):
    """(数据格式（节点1 节点2 权重）)
    Load the network data from a file and create an undirected graph.
    The file should have three columns: node1, node2, and weight (we ignore the weight).
    """
    df = pd.read_csv(file_path, sep=' ', header=None)  # 读取空格分隔的文件
    G = nx.Graph()
    G.add_edges_from(df.iloc[:, :2].values)  # 只使用前两列构建无向图
    return G

# def load_network(file_path):
#     """（数据格式（节点1,节点2））
#     Load the network data from a file and create an undirected graph.
#     The file should have two columns representing the edges of the network.
#     """
#     # Read the file as a DataFrame, assuming comma-separated values
#     df = pd.read_csv(file_path, sep=',', header=None)
#     G = nx.Graph()
#     G.add_edges_from(df.values)
#     return G

def calculate_degree_distribution(G, all_nodes):
    """
    Calculate the degree distribution of a network G for all_nodes.
    For nodes not present in the network, their degree is assumed to be 0.
    """
    degree_sequence = np.array([G.degree(n) if n in G else 0 for n in all_nodes])
    degree_distribution = degree_sequence / degree_sequence.sum()  # 归一化为概率分布
    return degree_distribution

def kl_divergence(P, Q):
    """
    Calculate the KL divergence between two probability distributions P and Q.
    """
    epsilon = 1e-10
    Q = np.where(Q == 0, epsilon, Q)  # 防止除以0
    P = np.where(P == 0, epsilon, P)  # 防止对数0
    kl_terms = P * np.log(P / Q)
    kl_div = np.sum(kl_terms)
    return kl_div

def main():
    # Load the two layers of the network
    # file_path_layer1 = './CS-5./1.txt'  # 替换为实际文件路径
    # file_path_layer2 = './CS-5./2.txt'  # 替换为实际文件路径
    # file_path_layer3 = './CS-5./3.txt'
    # # file_path_layer4 = './CS-5./4.txt'
    # # file_path_layer5 = './CS-5./5.txt'

    file_path_layer1 = 'mutilayer1.txt'
    file_path_layer2 = 'mutilayer2.txt'
    file_path_layer3 = 'mutilayer3.txt'

    G1 = load_network(file_path_layer1)
    G2 = load_network(file_path_layer2)
    G3 = load_network(file_path_layer3)
    # G4 = load_network(file_path_layer4)
    # G5 = load_network(file_path_layer5)

    # 假设 networks 是一个包含所有层网络的列表
    # networks = [G1, G2, G3, G4, G5]
    networks = [G1, G2, G3]

    # 获取所有网络层的节点并合并到一个集合中
    all_nodes = sorted(set().union(*[G.nodes for G in networks]))

    # 节点总数，用于归一化
    n = len(all_nodes)

    # 计算度分布
    P1= calculate_degree_distribution(G1, all_nodes)
    P2 = calculate_degree_distribution(G2, all_nodes)
    P3 = calculate_degree_distribution(G3, all_nodes)
    # P4 = calculate_degree_distribution(G4, all_nodes)
    # P5 = calculate_degree_distribution(G5, all_nodes)

    # 计算KL散度
    kl_12 = kl_divergence(P1, P2)
    kl_21 = kl_divergence(P2, P1)
    kl_13 = kl_divergence(P1, P3)
    kl_31 = kl_divergence(P3, P1)
    # kl_14 = kl_divergence(P1, P4)
    # kl_41 = kl_divergence(P4, P1)
    # kl_15 = kl_divergence(P1, P5)
    # kl_51 = kl_divergence(P5, P1)
    kl_23 = kl_divergence(P2, P3)
    kl_32 = kl_divergence(P3, P2)
    # kl_24 = kl_divergence(P2, P4)
    # kl_42 = kl_divergence(P4, P2)
    # kl_25 = kl_divergence(P2, P5)
    # kl_52 = kl_divergence(P5, P2)
    # kl_34 = kl_divergence(P3, P4)
    # kl_43 = kl_divergence(P4, P3)
    # kl_35 = kl_divergence(P3, P5)
    # kl_53 = kl_divergence(P5, P3)
    # kl_45 = kl_divergence(P4, P5)
    # kl_54 = kl_divergence(P5, P4)

    # # 最大值 KL 散度（你可以根据实际情况设定）
    # kl_max = max(kl_12, kl_21)

    # 使用 log-norm 归一化
    normalized_kl_12 = 1 / np.log2(2 + kl_12)
    normalized_kl_21 = 1 / np.log2(2 + kl_21)
    normalized_kl_13 = 1 / np.log2(2 + kl_13)
    normalized_kl_31 = 1 / np.log2(2 + kl_31)
    # normalized_kl_14 = 1 / np.log2(2 + kl_14)
    # normalized_kl_41 = 1 / np.log2(2 + kl_41)
    # normalized_kl_15 = 1 / np.log2(2 + kl_15)
    # normalized_kl_51 = 1 / np.log2(2 + kl_51)
    normalized_kl_23 = 1 / np.log2(2 + kl_23)
    normalized_kl_32 = 1 / np.log2(2 + kl_32)
    # normalized_kl_24 = 1 / np.log2(2 + kl_24)
    # normalized_kl_42 = 1 / np.log2(2 + kl_42)
    # normalized_kl_25 = 1 / np.log2(2 + kl_25)
    # normalized_kl_52 = 1 / np.log2(2 + kl_52)
    # normalized_kl_34 = 1 / np.log2(2 + kl_34)
    # normalized_kl_43 = 1 / np.log2(2 + kl_43)
    # normalized_kl_35 = 1 / np.log2(2 + kl_35)
    # normalized_kl_53 = 1 / np.log2(2 + kl_53)
    # normalized_kl_45 = 1 / np.log2(2 + kl_45)
    # normalized_kl_54 = 1 / np.log2(2 + kl_54)


    # 计算皮尔逊相关系数
    def calculate_pearson_correlation(P, Q):
        return pearsonr(P, Q)[0]  # 返回皮尔逊相关系数

        # 计算度分布的皮尔逊相关系数

    pearson_corr12 = calculate_pearson_correlation(P1, P2)
    pearson_corr21 = calculate_pearson_correlation(P2, P1)

    # 打印归一化后的KL散度
    print("*****打印计算的KL散度*****")
    print(f"KL_12: {kl_12}")
    print(f"KL_21: {kl_21}")
    print(f"KL_13: {kl_13}")
    print(f"KL_31: {kl_31}")
    # print(f"KL_14: {kl_14}")
    # print(f"KL_41: {kl_41}")
    # print(f"KL_15: {kl_15}")
    # print(f"KL_51: {kl_51}")
    print(f"KL_23: {kl_23}")
    print(f"KL_32: {kl_32}")
    # print(f"KL_24: {kl_24}")
    # print(f"KL_42: {kl_42}")
    # print(f"KL_25: {kl_25}")
    # print(f"KL_52: {kl_52}")
    # print(f"KL_34: {kl_34}")
    # print(f"KL_43: {kl_43}")
    # print(f"KL_35: {kl_35}")
    # print(f"KL_53: {kl_53}")
    # print(f"KL_45: {kl_45}")
    # print(f"KL_54: {kl_54}")
    print("*****打印归一化后不对称参数的*****")
    print(f"归一化的 KL_12: {normalized_kl_12}")
    print(f"归一化的 KL_21: {normalized_kl_21}")
    print(f"归一化的 KL_13: {normalized_kl_13}")
    print(f"归一化的 KL_31: {normalized_kl_31}")
    # print(f"归一化的 KL_14: {normalized_kl_14}")
    # print(f"归一化的 KL_41: {normalized_kl_41}")
    # print(f"归一化的 KL_15: {normalized_kl_15}")
    # print(f"归一化的 KL_51: {normalized_kl_51}")
    print(f"归一化的 KL_23: {normalized_kl_23}")
    print(f"归一化的 KL_32: {normalized_kl_32}")
    # print(f"归一化的 KL_24: {normalized_kl_24}")
    # print(f"归一化的 KL_42: {normalized_kl_42}")
    # print(f"归一化的 KL_25: {normalized_kl_25}")
    # print(f"归一化的 KL_52: {normalized_kl_52}")
    # print(f"归一化的 KL_34: {normalized_kl_34}")
    # print(f"归一化的 KL_43: {normalized_kl_43}")
    # print(f"归一化的 KL_35: {normalized_kl_35}")
    # print(f"归一化的 KL_53: {normalized_kl_53}")
    # print(f"归一化的 KL_45: {normalized_kl_45}")
    # print(f"归一化的 KL_54: {normalized_kl_54}")


    print(f"度分布的皮尔逊相关系数12: {pearson_corr12}")
    print(f"度分布的皮尔逊相关系数21: {pearson_corr21}")

    # print(P1)
    # print(P2)
if __name__ == "__main__":
    main()
