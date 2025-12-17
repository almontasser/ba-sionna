
# python train.py --require_gpu --channel_gen_device gpu \
#     --train_channels_outside_graph 1 -T 16 --epochs 200 --scenarios "UMi"
#     # ,UMa,RMa

# python train.py --require_gpu --channel_gen_device gpu \
#     --train_channels_outside_graph 1 -T 16 --epochs 10 --scenarios "UMi"
    # ,UMa,RMa

# # Use pre-generated channels -- obselete not working
# python train.py --require_gpu --channel_gen_device gpu \
#     --train_channels_outside_graph 1 -T 16 --epochs 100 --scenarios "UMi,UMa,RMa" --channel_dataset data/channels/train 

# Generate channels
python train.py --require_gpu --channel_gen_device gpu \
    --train_channels_outside_graph 1 -T 16 --epochs 50 --scenarios "UMi,UMa,RMa" --channel_cache_size 500 

python train.py --require_gpu --channel_gen_device gpu \
    --train_channels_outside_graph 1 -T 16 --epochs 50 --scenarios "UMi,UMa,RMa" --channel_cache_size 500 

python train.py --require_gpu --channel_gen_device gpu \
    --train_channels_outside_graph 1 -T 16 --epochs 50 --scenarios "UMi,UMa,RMa" --channel_cache_size 500 

python train.py --require_gpu --channel_gen_device gpu \
    --train_channels_outside_graph 1 -T 16 --epochs 50 --scenarios "UMi,UMa,RMa" --channel_cache_size 500 

python train.py --require_gpu --channel_gen_device gpu \
    --train_channels_outside_graph 1 -T 16 --epochs 50 --scenarios "UMi,UMa,RMa" --channel_cache_size 500 

python train.py --require_gpu --channel_gen_device gpu \
    --train_channels_outside_graph 1 -T 16 --epochs 50 --scenarios "UMi,UMa,RMa" --channel_cache_size 500 

python train.py --require_gpu --channel_gen_device gpu \
    --train_channels_outside_graph 1 -T 16 --epochs 50 --scenarios "UMi,UMa,RMa" --channel_cache_size 500 
