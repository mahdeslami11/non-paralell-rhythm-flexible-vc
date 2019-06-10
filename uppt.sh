export CUDA_VISIBLE_DEVICES=0
#python3 main.py --config="./config/config.yaml" --uppt --train --seed 2401 --A_id="p231" --B_id="p265"
python3 main.py --config="./config/config.yaml" --uppt --train --seed 2401 --A_id="all" --B_id="all"
