import os
import os.path as osp
import time
import sys
sys.path.append(os.path.abspath('.') + "/demo/libs/")
from track_uav import *
from segment import *
from recognise_uav import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 指定使用单卡

def main():
    output_dir = "./demo/output/OutputVideos/"
    os.makedirs(output_dir, exist_ok=True)
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    video_save_folder = osp.join(output_dir, timestamp)
    
    save_root = './demo/output/'

    gallery_video_path = "./demo/output/InputVideos/2.mp4"
    probe1_video_path  = "./demo/output/InputVideos/1.mp4"

    # tracking
    gallery_track_result = track(gallery_video_path, video_save_folder)
    probe1_track_result  = track(probe1_video_path, video_save_folder)


    gallery_video_name = gallery_video_path.split("/")[-1]
    gallery_video_name = save_root+'/try_seg/'+gallery_video_name.split(".")[0]

    probe1_video_name  = probe1_video_path.split("/")[-1]
    probe1_video_name  = save_root+'/try_seg/'+probe1_video_name.split(".")[0]
    

    exist = os.path.exists(gallery_video_name) and os.path.exists(probe1_video_name)
    
    print(exist)

    if exist:
        gallery_silhouette = getsil(gallery_video_path, save_root+'/try_seg/')
        probe1_silhouette  = getsil(probe1_video_path , save_root+'/try_seg/')

    else:
        gallery_silhouette = seg(gallery_video_path, gallery_track_result, save_root+'/try_seg/')
        probe1_silhouette  = seg(probe1_video_path , probe1_track_result , save_root+'/try_seg/')


    gallery_feat = extract_sil(gallery_silhouette, save_root+'/GaitFeatures/') 
    probe1_feat  = extract_sil(probe1_silhouette , save_root+'/GaitFeatures/')
    
    gallery_probe1_result = compare(probe1_feat, gallery_feat)

    writeresult(gallery_probe1_result, probe1_video_path, video_save_folder)

if __name__ == "__main__":
    main()
