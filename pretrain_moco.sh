lr=0.2
CUDA_VISIBLE_DEVICES=0,1 \
python main_moco.py \
  -a resnet50 \
  --lr ${lr} \
  --workers 2 \
  --batch-size 64 \
  --moco-k 65536 \
  --mixup --cos --mlp \
  --mixup-p 1.0 \
  --moco-t 0.2 \
  --amp-opt-level O1 \
  --save-dir "output/kuzikus/mocov2/moco-mixup-inverse-eric-epochs200/" \
  --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 \
  '/home/xiaochen/pretrain_dataset_256_seed11369'
