import torch
import numpy as np

def getemb(data):
    return data["inference_feat"]

def computedistence(x, y):
    distance = torch.sqrt(torch.sum(torch.square(x - y)))
    return distance

def compareid(data, dict, pid, threshold_value):
    probe_name = pid.split("-")[0]
    embs = getemb(data)
    min = threshold_value
    id = None
    dic={}
    for key in dict:
        if key == probe_name:
            continue
        for subject in dict[key]:
            for type in subject:
                for view in subject[type]:
                    value = subject[type][view]
                    distance = computedistence(embs["embeddings"],value)
                    gid = key + "-" + str(type)
                    gid_distance = (gid, distance)
                    dic[gid] = distance
                    if distance.float() < min:
                        id = gid
                        min = distance.float()
    dic_sort= sorted(dic.items(), key=lambda d:d[1], reverse = False)
    if id is None:
        print("############## no id #####################")
    return id, dic_sort


def comparefeat(embs, gallery_feat: dict, pid, threshold_value):
    #参数说明：embds：查询样本（probe）对应的特征嵌入
    #gallery_feat (dict)：一个字典，存储图库样本的特征
    #pid (str)：查询样本的标识符，表示查询样本的 ID。
    # threshold_value (int)：距离阈值，用于限制匹配的最远距离
    """Compares the distance between features

    Args:
        embs (Tensor): Embeddings of person with pid
        gallery_feat (dict): Dictionary of features from gallery
        pid (str): The id of person in probe
        threshold_value (int): Threshold
    Returns:
        id (str): The id in gallery
        dic_sort (dict): Recognition result sorting dictionary
    """
    probe_name = pid.split("-")[0] #通过 pid（查询样本的 ID）分割得到 probe_name，该值用于在图库中查找对应的样本。
    min = threshold_value #初始化最小距离为 threshold_value，即定义一个初始的较大距离，只有距离小于此阈值的匹配才会被接受。
    id = None
    dic={} #用于存储每个 gallery 特征与查询特征的距离。

    for key in gallery_feat:
        if key == probe_name:
            continue
        #遍历每个图库样本的特征
        for subject in gallery_feat[key]:
            #进一步遍历每个样本的不同类型和视角。
            for type in subject:
                for view in subject[type]:
                    #获取当前图库样本的特征值。
                    value = subject[type][view]
                    #计算距离
                    distance = computedistence(embs, value)
                    #生成当前图库样本的 ID，结合了图库样本的标识符和类型信息。
                    gid = key + "-" + str(type)
                    gid_distance = (gid, distance)
                    #将当前图库样本的 ID 和对应的距离存入字典 dic。
                    dic[gid] = distance
                    if distance.float() < min:
                        id = gid
                        min = distance.float()
    dic_sort= sorted(dic.items(), key=lambda d:d[1], reverse = False)
    if id is None:
        print("############## no id #####################")
    return id, dic_sort
    #dic_sort 存储了所有 gallery 特征与查询样本特征的距离，按照距离从小到大排序。
