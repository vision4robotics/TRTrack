# TRTrack: Boosting UAV Object Tracking with Voxel-based Trajectory-aware Reconstruction Training
## Abstract
Siamese network-based object tracking has remarkably promoted the automatic capability for highlymaneuvered unmanned aerial vehicles (UAVs). However, the leading-edge tracking framework often depends on template matching, making it trapped when facing multiple views of object in consecutive frames. Moreover, the general imagelevel pretrained backbone can overfit to holistic representations, causing the misalignment to learn object-level properties in UAV tracking. To tackle these issues, this work presents TRTrack, a comprehensive framework to fully exploit the stereoscopic representation for UAV tracking. Specifically, a novel pretraining paradigm method is proposed. Through trajectoryaware reconstruction training, the capability of the backbone to extract stereoscopic structure feature is strengthened without any parameter increment. Accordingly, an innovative hierarchical self-attention Transformer is proposed to capture the local detail information and global structure knowledge. For optimizing the correlation map, we proposed a novel spatial correlation refinement (SCR) module, which promotes the capability of modeling the long-range spatial dependencies. Comprehensive experiments on three UAV challenging benchmarks demonstrate that the proposed TRTrack achieves superior UAV tracking performance in both precision and efficiency. Quantitative tests in real-world settings fully prove the effectiveness of our work.. 
The demo videos are available at: https://www.bilibili.com/video/BV1PR4y1f74w/?vd_source=100fb1435454490a564488d3d32f4e88. 

<!-- ![Workflow of our tracker](https://github.com/vision4robotics/ResamplingNet/blob/main/images/workflow.jpg) -->
## About Code
### 1. Environment setup
This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2. Please install related libraries before running this code:

      pip install -r requirements.txt
<!-- ### 2. Test
Download pretrained model: [AFRTmodel](https://pan.baidu.com/s/1xXs60LeQehvCwKJo1zwzrg)(code: huat) and put it into `tools/snapshot` directory.

Download testing datasets and put them into `test_dataset` directory. If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit.git) to set test_dataset.

       python test.py 
	        --dataset UAV123                #dataset_name
	        --snapshot snapshot/AFRTmodel.pth  # tracker_name
	
The testing result will be saved in the `results/dataset_name/tracker_name` directory. -->
### 3. Train
#### Prepare training datasets

Download the datasets：

[VID](https://image-net.org/challenges/LSVRC/2017/)
 
[COCO](https://cocodataset.org/#home)

[GOT-10K](http://got-10k.aitestunion.com/downloads)

[LaSOT](http://vision.cs.stonybrook.edu/~lasot/)

#### Train a model

To train the TRT model, run `train.py` with the desired configs:

       cd tools
       python train.py

<!-- ### 4. Evaluation
We provide the tracking [results](https://pan.baidu.com/s/1d8P3O9V3I6jqDqgG2LG5Ng)(code: 6q8m) of UAV123@10fps, UAV123, and UAVTrack112_L. If you want to evaluate the tracker, please put those results into `results` directory.

        python eval.py 	                          \
	         --tracker_path ./results          \ # result path
	         --dataset UAV123                  \ # dataset_name
	         --tracker_prefix 'AFRTmodel'   # tracker_name -->
### 5. Contact
If you have any questions, please contact me.

Sihang Li

Email: <1752336@tongji.edu.cn>

## Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot.git). We would like to express our sincere thanks to the contributors.
