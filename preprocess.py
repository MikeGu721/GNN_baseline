'''
@author: Zhouhong Gu
@idea: Some preprocess methods
'''
import numpy as np
import gzh


def getAdj(cites_path, paper2id_path=None, id2paper_path=None):
    '''
    get a directed graph from cites path
    :param cites_path:
    :param paper2id_path:
    :param id2paper_path:
    :return:
    '''
    f = open(cites_path, encoding='utf-8')
    paper2id = {}
    id2paper = {}
    edges = []
    for line in f:
        a, b = line.strip().split('\t')
        for i in [a, b]:
            i_id = str(paper2id.get(i, len(paper2id)))
            paper2id[i] = i_id
            id2paper[i_id] = i
        edges.append([int(paper2id.get(a)), int(paper2id.get(b))])
    f.close()
    if id2paper_path:
        gzh.toJson(id2paper, id2paper_path)
    if paper2id_path:
        gzh.toJson(paper2id, paper2id_path)
    adj = np.zeros((len(paper2id), len(paper2id)))
    for a, b in edges:
        adj[a][b] = 1
        # if directed, delete next line
        adj[b][a] = 1
    return adj


def getContent(content_path,content_type=int):
    paper2content = {}
    f = open(content_path, encoding='utf-8')
    for line in f:
        line = line.strip().split('\t')
        paper_id = line[0]
        content = line[1:-1]
        paper2content[paper_id] = [content_type(i) for i in content]
    f.close()

    return paper2content


if __name__ == '__main__':
    from config import content_path, cites_path, paper2id_path, id2paper_path

    adj = getAdj(cites_path, paper2id_path, id2paper_path)
    paper2content = getContent(content_path)
    # print(adj.shape)
    print(paper2content)
