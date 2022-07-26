# Detectron2

    Mask RCNN, Pointrend, Faster RCNN and evaluation
    
    
# Dependencies:
    pip3 install -r requirements.txt
    
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{CUDA_VERSION}/{TORCH_VERSION}/index.html
    
    git clone --branch v0.6 https://github.com/facebookresearch/detectron2.git detectron2_repo
    
    pip install -e detectron2_repo  (If have to compile from source) (optional)
    
    Version used locally (Cuda 11.7) - 
    python3 -m pip install detectron2==0.6 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
