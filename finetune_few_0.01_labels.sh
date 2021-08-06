CUDA_ID=0
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${CUDA_ID} \
python main_finetune.py \
  -a resnet50 \
  --lr 0.01 \
  --workers 2 \
  --epochs 200 \
  --cos --mixup \
  --batch-size 256 \
  --pretrained '/home/xiaochen/pretrain_output/mocov2+cld/geo-color-shared-head-lr0.2-Lambda0.25-cld_t0.4-clusters32-NormNLP-epochs200/checkpoint_0149.pth.tar' \
  --save-dir 'output/mixup_0.01_fewshot/' \
  '/home/xiaochen/KWD-LT-0.01'