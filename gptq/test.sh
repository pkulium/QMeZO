git pull
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-13b c4 --wbits 2 --groupsize 1024 --save opt13-2bit.pt
# CUDA_VISIBLE_DEVICES=0 python -m pdb opt.py opt13-2bit.pt c4  
CUDA_VISIBLE_DEVICES=0 python -m pdb opt.py --load opt13-2bit.pt c4  