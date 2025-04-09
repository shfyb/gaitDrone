import os
import os.path as osp
import time
import sys
sys.path.append(os.path.abspath('.') + "/demo/libs/")
from track_uav import *
from segment_mmln import *
from recognise import *

def main():
    output_dir = "./demo/output/seg_Videos/"
    os.makedirs(output_dir, exist_ok=True)
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    video_save_folder = osp.join(output_dir, timestamp)
    
    save_root = './demo/output/'

    gallery_video_path = "./demo/output/InputVideos/2.mp4"

    # tracking
    gallery_track_result = track(gallery_video_path, video_save_folder)


    gallery_video_name = gallery_video_path.split("/")[-1]

    gallery_video_name = save_root+'/try_seg/'+gallery_video_name.split(".")[0]
    

    exist = os.path.exists(gallery_video_name)
    
    print(exist)

    if exist:
        gallery_silhouette = getsil(gallery_video_path, save_root+'/try_seg/')

    else:
        gallery_silhouette = seg(gallery_video_path, gallery_track_result, save_root+'/try_seg/')

    print("gallery_silhouette",gallery_silhouette)


if __name__ == "__main__":
    main()
