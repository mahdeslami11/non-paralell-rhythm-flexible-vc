import argparse
import yaml
import random
import numpy as np
import torch

from src.solver import PPR_Solver, PPTS_Solver

def train(s, args, config):
    while s.epoch < 100000:
        s.train()
        if s.epoch % 10 == 0:
            s.eval()
    return

def test(s, args, config):
    s.test()
    return

def main(args, config):
    if args.train:
        mode = 'train'
    elif args.test:
        mode = 'test'
    else:
        print("[Mode Error] Expecting --train or --test")

    if args.ppr:
        s = PPR_Solver(config, args, mode)
    elif args.ppts:
        s = PPTS_Solver(config, args, mode)
    elif args.uppt:
        s = UPPT_Solver(config, args, mode)
    else:
        print("[Model Error] Expecting --ppr or --ppts or --uppt")

    if args.train:
        train(s, args, config)
    else:
        test(s, args, config)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/Test on PPR/PPTS/UPPT')
    parser.add_argument('--config', type=str)
    parser.add_argument('--seed', default=0, type=int, required=False)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--train', action='store_true', default=False, dest='train')
    mode_group.add_argument('--test', action='store_true',default=False, dest='test')
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--ppr', action='store_true', default=False, dest='ppr')
    model_group.add_argument('--ppts', action='store_true',default=False, dest='ppts')
    model_group.add_argument('--uppt', action='store_true',default=False, dest='uppt')
    parser.add_argument('--spk_id', default=None, type=str, required=False)
    parser.add_argument('--A_id', default=None, type=str, required=False)
    parser.add_argument('--B_id', default=None, type=str, required=False)
    args = parser.parse_args()
    config = yaml.load(open(args.config))

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args, config)
