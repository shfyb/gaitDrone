import os
import cv2
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm

from fair_track_uav.lib.opts import opts  # FairMOT 配置
from fair_track_uav.lib.tracker.multitracker import JDETracker  # FairMOT 追踪器
from fair_track_uav.lib.tracking_utils.timer import Timer
from fair_track_uav.lib.tracking_utils.visualization import plot_tracking 
from fair_track_uav.lib.datasets.dataset.jde import *
import logging
track_cfgs = {  
    "model":{
        # "seg_model" : "./demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml",
        "ckpt" :    "./demo/checkpoints/bytetrack_model/bytetrack_x_mot17.pth.tar",# 1
        "exp_file": "./demo/checkpoints/bytetrack_model/yolox_x_mix_det.py", # 4
        
        "fair_ckpt": "./demo/checkpoints/fairmot_model/fairmot_dla34.pth"
    },
    "gait":{
        "dataset": "GREW",
    },
    "device": "gpu",
    "save_result": "True",
}

# 解析 FairMOT 配置
opt = opts().init()
opt.load_model = track_cfgs["model"]["fair_ckpt"]

# 加载 FairMOT 追踪器
def load_tracker():
    tracker = JDETracker(opt, frame_rate=30)
    return tracker

# 图像预处理（用于 FairMOT）
def prep_image(img, opt):
    img_size = (opt.input_w, opt.input_h)  # 使用 opt.input_w 和 opt.input_h
    img, ratio, padw, padh = letterbox(img, height=img_size[1], width=img_size[0])

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    return img.unsqueeze(0), img_size


def track(video_path, video_save_folder):
    frame_id = 0
    
    """Tracks person in the input video

    Args:
        video_path (Path): Path of input video
        video_save_folder (Path): Tracking video storage root path after processing
    Returns:
        track_results (dict): Track information
    """

    # # 获取视频的基本信息：宽度、高度、帧数和FPS（帧率）
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    tracker = load_tracker()  # 加载 FairMOT 追踪器
    timer = Timer() ## 计时器，用于计算处理每帧所需时间

    #创建保存视频文件的路径
    os.makedirs(video_save_folder, exist_ok=True)
    save_video_name = video_path.split("/")[-1] # 获取视频文件名（不含路径）
    save_video_path = osp.join(video_save_folder, save_video_name)  # 组合成完整路径
    print(f"video save_path is {save_video_path}") # 打印保存路径
    
    # 创建视频写入对象，保存跟踪后的视频
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    save_video_name = save_video_name.split(".")[0]  # 移除文件扩展名，仅保留视频名称
    results = []# 存储跟踪结果（文本格式）
    track_results={}# 存储每帧的跟踪信息（字典格式）
    mark = True
    diff = 0
    #通过循环遍历视频的每一帧，读取视频帧。tqdm 是一个用于显示进度条的库。
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()
        if not ret_val:
            break

        timer.tic()
        blobs, img0 = prep_image(frame, opt)  # 预处理图像
          # 将输入数据移动到 GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        blobs = blobs.to(device)

        online_targets = tracker.update(blobs, frame)  # 进行目标跟踪

        online_tlwhs, online_ids = [], []  # 存储当前帧的目标位置信息（tlwh）和 ID
        for t in online_targets:

            tlwh = t.tlwh # 获取目标的边界框信息 (top-left x, y, width, height)
            tid = t.track_id # 获取目标 ID
            #mark = True 表示这是第一次处理 tid，当 mark == True 时，执行：
            if mark:
                #mark = False 确保 diff 只计算一次，不会在后续的 tid 处理过程中重复执行。
                mark = False
                #计算 diff = tid - 1，即当前 tid 相对于 1 的偏移量。
                diff = tid - 1
                #这样可以 让 ID 重新从 1 开始，即第一个 tid 变成 1，后面的 tid 依次调整。
            tid = tid - diff

            online_tlwhs.append(tlwh)
            online_ids.append(tid)
             # 将跟踪结果存入字典，按照 frame_id 进行存储
            if frame_id not in track_results:
                track_results[frame_id] = []
            track_results[frame_id].append([tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])
            # 追加文本格式的跟踪结果
            results.append(
                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},-1,-1,-1\n"
            )
        timer.toc()# 计时结束
        # 绘制跟踪结果，将目标框绘制在图像上
        online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id, fps=1. / timer.average_time)
        vid_writer.write(online_im)# 将绘制后的帧写入视频
        
        frame_id += 1 # 递增帧编号

     # 如果配置中要求保存结果，则将跟踪数据写入文本文件
    if track_cfgs["save_result"] == "True":
        res_file = osp.join(video_save_folder, f"{save_video_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)

    return track_results  # 返回跟踪结果：frame_id, track_id, x, y, w, h, -1, -1, -1

    '''
    frame_id：当前帧的编号，从 0 开始。
    track_id：目标 ID，每个被 FairMOT 追踪的目标会被分配一个唯一的 ID。
    x：目标边界框左上角的 x 坐标（单位：像素）。
    y：目标边界框左上角的 y 坐标（单位：像素）。
    w：目标边界框的宽度（单位：像素）。
    h：目标边界框的高度（单位：像素）。
    -1：通常是置信度分数，但在此代码中始终为 -1，表示未使用。
    -1：类别 ID，FairMOT 默认用于行人跟踪，因此未指定类别，设为 -1。
    -1：可忽略字段，FairMOT 的标准格式中占位。
    
    '''

def writeresult(pgdict, video_path, video_save_folder):
    """Writes the recognition result back into the video (FairMOT 版本)

    Args:
        pgdict (dict): 探针ID到图库ID的映射字典
        video_path (Path): 输入视频路径
        video_save_folder (Path): 处理后的视频保存路径
        opt: 图像预处理配置参数
        track_cfgs: 跟踪配置参数
    """
    frame_id = 0
    id_map = {}  # 原始 Track ID 到新 ID 的映射
    current_id = 1  # 新 ID 从 1 开始
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    # 获取视频参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化跟踪器和保存路径
    tracker = load_tracker()  # 加载 FairMOT 跟踪器
    os.makedirs(video_save_folder, exist_ok=True)
    video_name = osp.splitext(osp.basename(video_path))[0]
    first_key = next(iter(pgdict))
    gallery_name = pgdict[first_key].split("-")[0]
    save_video_name = f"G-{gallery_name}_P-{video_name}.mp4"
    save_video_path = osp.join(video_save_folder, save_video_name)
    
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    results = []
    timer = Timer()

    try:
        for _ in tqdm(range(frame_count)):
            ret_val, frame = cap.read()
            if not ret_val:
                break

            timer.tic()
            # FairMOT 预处理
            blobs, img0 = prep_image(frame, opt)
            blobs = blobs.to(device)
            
            # 执行跟踪
            online_targets = tracker.update(blobs, frame)

            online_tlwhs = []
            online_ids = []
            online_colors = []
            
            for t in online_targets:
                tlwh = t.tlwh
                original_tid = t.track_id

                # 更新 ID 映射表
                if original_tid not in id_map:
                    id_map[original_tid] = current_id
                    current_id += 1
                track_id = id_map[original_tid]

                # 生成探针ID并查询映射表
                pid = f"{video_name}-{track_id:03d}"

                #tid = pgdict.get(pid, "Unknown")  # 处理未映射的ID

                tid_str = pgdict.get(pid, "Unknown")  # 处理未映射的ID

                if tid_str != "Unknown":
                    try:
                        tid = int(tid_str.replace("-", ""))  # 移除非数字字符后转换
                    except:
                        tid = 0  # 转换失败时默认值
                else:
                    tid = 0
                

                # 颜色生成逻辑
                color_id = tid % 255  # 确保颜色值在0-255范围内

                # 过滤无效框
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > 10 and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_colors.append(color_id)
                    
                    # 记录结果
                    results.append(
                        f"{frame_id},{tid_str},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},-1,-1,-1\n"
                    )

            timer.toc()
            
            # 可视化跟踪结果
            online_im = plot_tracking(
                frame, 
                online_tlwhs,
                online_ids,
                frame_id=frame_id,
                fps=1. / timer.average_time
            )
            
            if track_cfgs.get("save_result", False):
                vid_writer.write(online_im)
            
            frame_id += 1

    finally:
        cap.release()
        vid_writer.release()

    # 保存文本结果
    if track_cfgs.get("save_result", False):
        res_file = osp.join(video_save_folder, f"{video_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)

    print(f"处理完成，结果保存至: {save_video_path}")