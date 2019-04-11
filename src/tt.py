import yaml
import solver

config = yaml.load(open('../config/config.yaml'))
s = solver.PPR_Solver(config, None)

s.build_model()
dl=s.get_dataset()

for k in dl:
    print(k)
