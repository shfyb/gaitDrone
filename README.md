
## All-in-One-Gait-UAV

## Folder Structure
```
All-in-One-Gait-UAV
   |——————checkpoints
   |        └——————fairmot_model
   |        └——————gait_model
   |        └——————seg_model
   └——————libs
   └——————output


checkpoints
   |——————bytetrack_model
   |        └——————fairmot_dla34.pth
   |
   └——————gait_model
   |        └——————GaitBase_DroneGait2_200-60000.pt
   └——————seg_model
            └——————human_pp_humansegv2_mobile_192x192_inference_model_with_softmax
```

## Detect&Track
  libs/fair_trcak_uav

  track_uav.py
## Segment
  pp-humanseg v2 : libs/paddle 

  segment.py
## Gait-UAV
  Agg-NET : libs/opengait
  
  recognise_uaav.py
  
## Install
   pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

   pip install -r requirements.txt
### Then
   Fairmot use DCNV2_pytorch in their backbone network.
   
    git clone https://github.com/lbin/DCNv2.git
    cd DCNv2
    ./make.sh
## Pretrained Weights of Backbones
   Fairmot [[baidu()]]()

   segment 
   ```
   wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip
   ```
   Gait [[baidu()]]()
## Run
   ```
   python main.py
   
   ```
## Results
   ```
output
   └——————GaitFeatures: This stores the corresponding gait features
   └——————GaitSilhouette: This stores the corresponding gait silhouette images
   └——————InputVideos: This is the folder where the input videos are put
   |       └——————1.mp4
   |       └——————2.mp4
   └——————OutputVideos
           └——————{timestamp}
                   └——————1.mp4
                   └——————G-2_P-1.mp4
```