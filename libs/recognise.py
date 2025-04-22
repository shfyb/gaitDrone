import os
import os.path as osp
import pickle
import sys
# import shutil

root = os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
sys.path.append(root)
from opengait.utils import config_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
from loguru import logger
import model.baselineDemo as baselineDemo
import gait_compare as gc

recognise_cfgs = {  
    "gaitmodel":{
        "model_type": "BaselineDemo",
        # "cfg_path": "./configs/baseline/baseline_GREW.yaml",
        "cfg_path": "./configs/gaitbase/gaitbase_da_gait3d.yaml",
    },
}

#通过传递模型类型 (model_type) 和配置文件路径 (cfg_path)
def loadModel(model_type, cfg_path):

    #动态获取 baselineDemo 类中名为 model_type：BaselineDemo 的属性或方法，并赋值给变量 Model
    Model = getattr(baselineDemo, model_type)

    cfgs = config_loader(cfg_path)

    model = Model(cfgs, training=False)

    return model


def gait_sil(sils, embs_save_path):
    """Gets the features.

    Args:
        sils (list): List of Tuple (seqs, labs, typs, vies, seqL)
        embs_save_path (Path): Output path.
    Returns:
        feats (dict): Dictionary of features
    """
    gaitmodel = loadModel(**recognise_cfgs["gaitmodel"])

    gaitmodel.requires_grad_(False)
    #设置模型为评估模式（eval）
    gaitmodel.eval()
    #用于存储最终提取的步态特征
    feats = {}

    for inputs in sils:
        #获取输入数据
        ipts = gaitmodel.inputs_pretreament(inputs)
        
        id = inputs[1][0]
        if id not in feats:
            feats[id] = []
        type = inputs[2][0] 
        view = inputs[3][0]

        embs_pkl_path = "{}/{}/{}/{}".format(embs_save_path, id, type, view)
        if not os.path.exists(embs_pkl_path):
            os.makedirs(embs_pkl_path)
        embs_pkl_name = "{}/{}.pkl".format(embs_pkl_path, inputs[3][0])

        #调用模型的 forward 方法，将预处理后的步态序列 ipts 输入到模型中
        #retval：模型的预测值（通常与任务相关）。embs：模型提取的特征向量（嵌入特征）。
        retval, embs = gaitmodel.forward(ipts)

        pkl = open(embs_pkl_name, 'wb')
        pickle.dump(embs, pkl)
        feat = {}
        feat[type] = {}
        feat[type][view] = embs
        feats[id].append(feat)        
    return feats    

def gaitfeat_compare(probe_feat:dict, gallery_feat:dict):
    """Compares the feature between probe and gallery

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
    Returns:
        pg_dicts (dict): The id of probe corresponds to the id of gallery
    """
    #获取 probe 特征的键（即 probe 的标识符），然后取第一个作为 probe（假设这里只有一个 probe）
    item = list(probe_feat.keys())
    probe = item[0]
    #pg_dict 用于存储每个 probe 与最匹配的 gallery 的 ID；
    # pg_dicts 存储更多的详细信息（如匹配的 ID 字典）。
    pg_dict = {}
    pg_dicts = {}
    #遍历 probe_feat[probe] 中的每一个元素
    for inputs in probe_feat[probe]:
        number = list(inputs.keys())[0]
        probeid = probe + "-" + number

        galleryid, idsdict = gc.comparefeat(inputs[number]['undefined'], gallery_feat, probeid, 100)
        #galleryid与当前 probe 特征最匹配的 gallery ID。
        #idsdict：详细的匹配信息，可能包含与多个 gallery 特征的匹配程度。
        pg_dict[probeid] = galleryid
        pg_dicts[probeid] = idsdict
    # print("=================== pg_dicts ===================")
    # print(pg_dicts)
    return pg_dict

def extract_sil(sil, save_path):
    """Gets the features.

    Args:
        sils (list): List of Tuple (seqs, labs, typs, vies, seqL)
        save_path (Path): Output path.
    Returns:
        video_feats (dict): Dictionary of features from the video
    """
    logger.info("begin extracting")
    video_feat = gait_sil(sil, save_path)
    logger.info("extract Done")
    return video_feat


def compare(probe_feat, gallery_feat):
    """Recognizes  the features between probe and gallery

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
    Returns:
        pgdict (dict): The id of probe corresponds to the id of gallery
    """
    logger.info("begin recognising")
    pgdict = gaitfeat_compare(probe_feat, gallery_feat)
    logger.info("recognise Done")

    print("================= probe - gallery ===================")
    print(pgdict)

    return pgdict