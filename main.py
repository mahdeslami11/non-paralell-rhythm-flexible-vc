import yaml
from src.solver import PPR_Solver, PPTS_Solver

config = yaml.load(open('./config/config.yaml'))
s = PPR_Solver(config, None, 'train')
while s.epoch < 1000:
    s.train()
    if s.epoch % 10 == 0:
        s.eval()
