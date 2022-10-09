conda create -n ENV python=3.7.4
conda activate ENV
pip install torch===1.8.0+cu101 torchvision===0.9.0+cu101 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple