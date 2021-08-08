lr=0.2
CUDA_VISIBLE_DEVICES=0,1 \
python main_moco.py \
  -a resnet50 \
  --lr ${lr} \
  --workers 2 \
  --batch-size 64 \
  --moco-k 4096 \
  --bimoco --mixup --cos --mlp --rui --replace \
  --bimoco-gamma 0.5 \
  --mixup-p 0.3 \
  --moco-t 0.2 \
  --amp-opt-level O1 \
  --knn-data "/home/xiaochen/KWD-LT-0.1" \
  --save-dir "output/kuzikus/mocov2/bimoco-mixup0.3-rui-replace-epochs200/" \
  --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 \
  '/home/xiaochen/pretrain_dataset_256_seed11369'
