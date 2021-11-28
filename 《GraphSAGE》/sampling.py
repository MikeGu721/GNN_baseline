'''
为了实现高效的采样，可以将节点及其邻居存放在一起，维护一个节点与其邻居对应关系的表——neighbor_table
'''

import numpy as np

def sampling(src_nodes, sample_num, neighbor_table):
    '''
    根据源节点采样指定数量的邻居节点，注意使用有放回的采样
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
    :param src_nodes: 源节点列表
    :param sample_num: 需要采样的节点数
    :param neighbor_table: 节点到其邻居节点的映射表
    :return: 采样结果构成的列表
    '''
    results = []
    for sid in src_nodes:
        res = np.random.choice(neighbor_table[sid], size=(sample_num, ))
        results.append(res)
    return np.asarray(results).flatten()

def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    '''
    根据源节点进行多阶采样
    :param src_nodes: 源节点id
    :param sample_nums: 每一阶需要采样的个数
    :param neighbor_table: 节点到邻居节点的映射
    :return: 每一阶采样的结果
    '''
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_reulst = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_reulst)
    return sampling_result