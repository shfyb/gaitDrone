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

track_cfgs = {  
    "model":{
        # "seg_model" : "./demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml",
        "ckpt" :    "./demo/checkpoints/bytetrack_model/bytetrack_x_mot17.pth.tar",# 1
        "exp_file": "./demo/checkpoints/bytetrack_model/yolox_x_mix_det.py", # 4
    },
    "gait":{
        "dataset": "GREW",
    },
    "device": "gpu",
    "save_result": "True",
}

#为目标 ID 分配不同颜色。
colors = [(255,0,0),(0,255,0),(0,0,255),(0,0,0)]
def get_color(idx):

    if idx<=4:
        color = colors[idx-1]
    else:
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color
'''
加载 YOLOX 模型，并从 ckpt 文件中读取预训练权重。
融合 BN 层 以提升推理速度。
使用 half() 进行 FP16 加速。

'''
def loadckpt(exp):
    device = torch.device("cuda" if track_cfgs["device"] == "gpu" else "cpu")
    model = exp.get_model().to(device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()
    ckpt_file = track_cfgs["model"]["ckpt"]
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    logger.info("\tFusing model...")
    model = fuse_model(model)
    model = model.half()
    return model

exp = get_exp(track_cfgs["model"]["exp_file"], None)
model = loadckpt(exp)

def track(video_path, video_save_folder):
    """Tracks person in the input video

    Args:
        video_path (Path): Path of input video
        video_save_folder (Path): Tracking video storage root path after processing
    Returns:
        track_results (dict): Track information
    """
    
    #初始化设备和预测器
    trt_file = None
    decoder = None
    device = torch.device("cuda" if track_cfgs["device"] == "gpu" else "cpu")
    predictor = Predictor(model, exp, trt_file, decoder, device, True)

    #读取视频文件，使用 cv2.VideoCapture 打开视频文件并获取视频的一些基本信息，如宽度、高度、总帧数和帧率。
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #初始化跟踪器，
    #使用 BYTETracker 初始化跟踪器，并设定帧率。
    #BYTETracker 是一种常用于多目标跟踪的算法，能够在视频中跟踪多个物体。
    #timer 用于计时和评估每帧处理的时间。
    tracker = BYTETracker(frame_rate=30)

    timer = Timer()
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    #创建保存视频文件的路径
    os.makedirs(video_save_folder, exist_ok=True)
    save_video_name = video_path.split("/")[-1]
    save_video_path = osp.join(video_save_folder, save_video_name)
    print(f"video save_path is {save_video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    save_video_name = save_video_name.split(".")[0]
    results = []
    track_results={}
    mark = True
    diff = 0

    #通过循环遍历视频的每一帧，读取视频帧。tqdm 是一个用于显示进度条的库。
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()

        if ret_val:
            #使用 predictor 对当前帧进行目标检测，得到检测结果（outputs）。
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                #然后使用 BYTETracker 更新跟踪目标的位置。outputs[0] 是当前帧检测到的目标框（如人的位置）。
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                #提取跟踪结果中的目标信息，如目标的位置（tlwh，左上角坐标和宽高）、ID 和置信度（score）。
                
                online_tlwhs = []
                online_ids = []
                online_scores = []

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    
                    if mark:
                        mark = False
                        diff = tid - 1
                    tid = tid - diff

                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        if frame_id not in track_results:
                            track_results[frame_id] = []
                        #如果当前帧中有目标跟踪信息，则将目标的跟踪结果添加到 track_results 字典中，按帧存储。
                        track_results[frame_id].append([tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                #将跟踪框绘制到当前帧上，plot_tracking 函数将显示目标的位置和 ID。
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
                
            if track_cfgs["save_result"] == "True":
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    #将每一帧的跟踪信息（目标ID、位置、置信度等）保存为文本文件。
    if track_cfgs["save_result"] == "True":
        res_file = osp.join(video_save_folder, f"{save_video_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

    return track_results 

def writeresult(pgdict, video_path, video_save_folder):
    """Writes the recognition result back into the video

    Args:
        pgdict (dict): The id of probe corresponds to the id of gallery
        video_path (Path): Path of input video
        video_save_folder (Path): Tracking video storage root path after processing
    """
    device = torch.device("cuda" if track_cfgs["device"] == "gpu" else "cpu")
    trt_file = None
    decoder = None
    predictor = Predictor(model, exp, trt_file, decoder, device, True)
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(video_save_folder, exist_ok=True)
    video_name = video_path.split("/")[-1]
    first_key = next(iter(pgdict))
    gallery_name = pgdict[first_key].split("-")[0]
    probe_name = video_name
    # save_video_path = save_video_name.split(".")[0]+ "-After.mp4"
    save_video_name = "G-{}_P-{}".format(gallery_name, probe_name)
    save_video_path = osp.join(video_save_folder, save_video_name)
    
    print(f"video save_path is {save_video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    video_name = video_name.split(".")[0]

    tracker = BYTETracker(frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    mark = True
    diff = 0
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_colors = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    if mark:
                        mark = False
                        diff = t.track_id - 1
                    track_id = t.track_id - diff

                    pid = "{}-{:03d}".format(video_name, track_id)
                    tid = pgdict[pid]
                    # demo
                    colorid = int(tid.split("-")[1])
                    # colorid = track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_colors.append(colorid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_track(
                    img_info['raw_img'], online_tlwhs, online_ids, online_colors, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if track_cfgs["save_result"] == "True":
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    # if track_cfgs["save_result"] == "True":
    #     txtfile = "{}-{}".format(save_video_name, "After.txt")
    #     res_file = osp.join(video_save_folder, txtfile)
    #     with open(res_file, 'w') as f:
    #         f.writelines(results)
    #     logger.info(f"save results to {res_file}")
