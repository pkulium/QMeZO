git pull
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-13b c4 --wbits 2 --groupsize 1024 --save opt13-2bit.pt
CUDA_VISIBLE_DEVICES=0 python opt.py --model facebook/opt-13b --dataset c4 --load opt13-2bit.pt  --benchmark 128