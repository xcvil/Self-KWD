lr=0.06
CUDA_VISIBLE_DEVICES=0 \
python main_cifar.py \
  -a resnet18 \
  --lr ${lr} \
  --workers 2 \
  --batch-size 64 \
  --moco-k 4096 \
  --bimoco --mixup --cos --mlp --rui --replace \
  --bimoco-gamma 0.5 \
  --mixup-p 0.3 \
  --moco-t 0.1 \
  --moco-m 0.99 \
  --wd 0.0005\
  --amp-opt-level O1 \
  --save-dir "output/kuzikus/mocov2/cifar-bimoco-mixup0.3-rui-replace-epochs200/" \
  --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0