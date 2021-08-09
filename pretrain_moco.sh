lr=0.2
CUDA_VISIBLE_DEVICES=0 \
python main_moco.py \
  -a resnet50 \
  --lr ${lr} \
  --workers 2 \
  --batch-size 64 \
  --moco-k 4096 \
  --cos --mlp \
  -knn-t 100 \
  --moco-t 0.2 \
  --amp-opt-level O1 \
  --knn-data "/home/xiaochen/KWD-LT" \
  --save-dir "output/kuzikus/mocov2/moco-v2-epochs200-LT/" \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  '/home/xiaochen/KWD-LT/train'