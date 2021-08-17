lr=0.06
CUDA_VISIBLE_DEVICES=0 \
python main_cifar.py \
  -a resnet18 \
  --lr ${lr} \
  --workers 2 \
  --batch-size 512 \
  --epochs 800 \
  --moco-k 4096 \
  --bimoco --mixup --cos --mlp --rui --replace \
  --bimoco-gamma 0.9 \
  --mixup-p 0.3 \
  --moco-t 0.1 \
  --moco-m 0.99 \
  --wd 0.0005 \
  --knn-k 200 \
  --knn-t 0.1 \
  --amp-opt-level O1 \
  --save-dir "output/kuzikus/mocov2/cifar-bimoco-gamma0.9-mixup0.3-rui-replace-epochs200/" \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0