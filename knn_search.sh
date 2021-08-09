CUDA_VISIBLE_DEVICES=0 \
python knn_search.py \
  -a resnet50 \
  --workers 2 \
  --resume '/home/xiaochen/journal_output/bimoco-mixup0.3-rui-replace-epochs200-LT/checkpoint_0199.pth.tar' \
  --save-dir "output/features/" \
  --bimoco --mlp \
  --moco-k 4096 \
  --moco-t 0.1 \
  --moco-m 0.99 \
  --wd 0.0005\
  --amp-opt-level O1 \
  --knn-data "/home/xiaochen/KWD-LT-0.1" \
  --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0