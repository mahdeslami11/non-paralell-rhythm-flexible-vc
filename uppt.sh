export CUDA_VISIBLE_DEVICES=0
python3 main.py --config="./config/config.yaml" --uppt --train --seed 343 --A_id="all" --B_id="all" --pre_train
