lr=0.1
CUDA_VISIBLE_DEVICES=0 \
python main_moco.py \
  -a resnet50 \
  --lr ${lr} \
  --workers 2 \
  --batch-size 64 \
  --moco-k 4096 \
  --bimoco --cos --mlp \
  --bimoco-gamma 0.9 \
  --moco-t 0.2 \
  --knn-k 20 \
  --knn-t 0.02 \
  --amp-opt-level O1 \
  --knn-data "/home/xiaochen/KWD-LT-0.1" \
  --save-dir "output/kuzikus/mocov2/bimoco-gamma0.9-geo-color-epochs200-LT/" \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  '/home/xiaochen/KWD-LT/train'
