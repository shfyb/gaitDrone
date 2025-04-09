import os 
import os.path as osp
import sys
import cv2
from pathlib import Path
import shutil
import torch
import math
import numpy as np
from tqdm import tqdm

from tracking_utils.predictor import Predictor
from yolox.utils import fuse_model, get_model_info
from loguru import logger
from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
from tracking_utils.visualize import plot_tracking, plot_track
from pretreatment import pretreat, imgs2inputs
sys.path.append((os.path.dirname(os.path.abspath(__file__) )) + "/paddle/")
from seg_demo import seg_image
from yolox.exp import get_exp

#定义了轮廓分割模型和数据集的一些配置信息。
seg_cfgs = {  
    "model":{
        "seg_model" : "./demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml",
    },
    "gait":{
        "dataset": "GREW",
    }
}

#通过视频的跟踪结果（track_result），根据跟踪框裁剪出每一帧图像中的人物轮廓，并保存到指定目录。
def imageflow_demo(video_path, track_result, sil_save_path):
    """Cuts the video image according to the tracking result to obtain the silhouette

    Args:
        video_path (Path): Path of input video
        track_result (dict): Track information
        sil_save_path (Path): The root directory where the silhouette is stored
    Returns:
        Path: The directory of silhouette
    """

    #获取视频的宽度、高度、帧数和帧率。
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    #提取视频文件的名称，并去掉文件扩展名。results 用于存储处理结果，ids 存储所有的跟踪帧 ID。
    save_video_name = video_path.split("/")[-1]
    save_video_name = save_video_name.split(".")[0]
    results = []
    ids = list(track_result.keys())


    #遍历视频的每一帧并读取。frame_id % 4 == 0 表示每隔 4 帧处理一次。
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()
        if ret_val:
            if frame_id in ids and frame_id%4==0:

                # track_result[frame_id] 是当前帧的跟踪信息，
                # 其中每个元素是一个 tidxywh，表示目标的 track_id 和目标的边界框坐标（x, y, width, height）。
                for tidxywh in track_result[frame_id]:
                    tid = tidxywh[0]
                    tidstr = "{:03d}".format(tid)
                    #为每个跟踪目标创建一个保存路径 savesil_path。
                    savesil_path = osp.join(sil_save_path, save_video_name, tidstr, "undefined")
                    #提取每个目标的x，y坐标和宽度高度
                    x = tidxywh[1]
                    y = tidxywh[2]
                    width = tidxywh[3]
                    height = tidxywh[4]

                    #对每个目标的边界框进行扩展（向四周扩展 10%），以确保在裁剪时捕捉到更多的区域。
                    x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
                    w, h = x2 - x1, y2 - y1
                    x1_new = max(0, int(x1 - 0.1 * w))
                    x2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(x2 + 0.1 * w))
                    y1_new = max(0, int(y1 - 0.1 * h))
                    y2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(y2 + 0.1 * h))
                    new_w = x2_new - x1_new
                    new_h = y2_new - y1_new
                    tmp = frame[y1_new: y2_new, x1_new: x2_new, :]

                    # 创建一个新的白色背景图像 tmp_new，将目标区域填充到其中，以保证所有提取的目标图像是正方形，并按比例调整。
                    save_name = "{:03d}-{:03d}.png".format(tid, frame_id)
                    side = max(new_w,new_h)
                    tmp_new = [[[255,255,255]]*side]*side
                    tmp_new = np.array(tmp_new)
                    width = math.floor((side-new_w)/2)
                    height = math.floor((side-new_h)/2)
                    tmp_new[int(height):int(height+new_h),int(width):int(width+new_w),:] = tmp
                    tmp_new = tmp_new.astype(np.uint8)

                    #将目标区域调整为 192x192 的尺寸
                    tmp = cv2.resize(tmp_new,(192,192))

                    #调用 seg_image 函数进行分割
                    seg_image(tmp, seg_cfgs["model"]["seg_model"], save_name, savesil_path)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    return Path(sil_save_path, save_video_name)

def seg(video_path, track_result, sil_save_path):
    """Cuts the video image according to the tracking result to obtain the silhouette

    Args:
        video_path (Path): Path of input video
        track_result (Path): Track information
        sil_save_path (Path): The root directory where the silhouette is stored
    Returns:
        inputs (list): List of Tuple (seqs, labs, typs, vies, seqL) 
    """
    # imageflow_demo 函数用于分割
    sil_save_path = imageflow_demo(video_path, track_result, sil_save_path)

    #imgs2inputs 函数将生成的轮廓图像转换为输入数据，准备进一步处理
    inputs = imgs2inputs(Path(sil_save_path), 64, False, seg_cfgs["gait"]["dataset"])
    return inputs

def getsil(video_path, sil_save_path):
    sil_save_name = video_path.split("/")[-1]

    inputs = imgs2inputs(Path(sil_save_path, sil_save_name.split(".")[0]), 
                64, False, seg_cfgs["gait"]["dataset"])
    
    return inputs
