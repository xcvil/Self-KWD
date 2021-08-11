CUDA_VISIBLE_DEVICES=0 \
python knn_search.py \
  -a resnet50 \
  --resume '/home/xiaochen/Self-KWD/output/kuzikus/mocov2/bimoco-gamma0.9-mixup0.3-rui-replace-epochs200-LT/checkpoint_0199.pth.tar' \
  --save-dir "output/features/" \
  --bimoco --mlp \
  --moco-k 4096 \
  --knn-data "/home/xiaochen/KWD-LT-0.1"