git pull
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --wbits 2 --groupsize 1024  --trits 1 --save opt1.3-2bit.pt
# CUDA_VISIBLE_DEVICES=0 python opt.py --model facebook/opt-1.3b --dataset c4 --load opt13-2bit.pt --trits True