python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="10.17.2.193" --master_port=12345 dinov2/train/train.py \
    --config-file=dinov2/configs/train/vitl16_short.yaml \
    --output-dir=output_dir/test1 \
    train.dataset_path=ImageNet:split=TRAIN:root=/medai/Test/Dinov2/imagenet_1k/process_dir/data:extra=/medai/Test/Dinov2/imagenet_1k/process_dir/extra_data

