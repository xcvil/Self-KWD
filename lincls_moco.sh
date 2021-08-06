CUDA_ID=0,1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_ID} \
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --workers 2 \
  --batch-size 256 \
  --pretrained '/home/xiaochen/moco-geo/mt_codebase/KMoCo/output/kuzikus/mocov2/moco-geo-epochs200/checkpoint_0069.pth.tar' \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  '/home/xiaochen/KWD'