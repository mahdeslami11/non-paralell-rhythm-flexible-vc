import yaml
import solver

config = yaml.load(open('../config/config.yaml'))
s = solver.PPR_Solver(config, None, 'train')
while s.epoch < 1000:
    s.train()
    if s.epoch % 10 == 0:
        s.eval()
