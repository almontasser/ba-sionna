
python train.py --require_gpu --channel_gen_device gpu \
    --train_channels_outside_graph 1 -T 16 --epochs 100 --scenarios "UMi,UMa,RMa"