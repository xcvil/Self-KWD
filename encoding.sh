CUDA_VISIBLE_DEVICES=0,1 \
python main_encoding.py \
  -a resnet50 \
  --workers 2 \
  --pretrained '/home/xiaochen/pretrain_output/moco-geo-color-shared-head-epochs200/checkpoint_0149.pth.tar' \
  --save-dir "output/features/" \
  '/home/xiaochen/KWD/test'
