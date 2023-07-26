from plus_state_compiling.stimFunctions import stim_functions as stim
from IPython.display import display
import Codes.reed_muller as rm
import Codes.surface_code as sc
import numpy as np
import sinter

params = {
            # Units are microseconds.
            'storage_T1': 3000000000, # 3ms
            'storage_T2': 3000000000, # 3ms
            'compute_T1': 100000000, # 100us
            'compute_T2': 100000000, # 100us
            'gate2_err': 0,
            'time_2q': .1, # 100ns
            'readout_err': 0,
            'time_measurement_compute': 1, # 1us
            'reset_err': 0,
            'time_reset_compute': 1.04, # 1.04us
            'compute_1q_err': 0,
            'time_1q_compute': .04, # 40ns
            'err_save_load': 0,
            'time_save_load': .1, # 100ns
        }

# Checks for Reed-Muller code:
X_checks, Z_checks = rm.reed_muller_checks()
print("X_checks: ", X_checks)
print("Z_checks: ", Z_checks)
checks = X_checks + Z_checks

num_rounds = 1
# Checks for (Lx, Ly) surface code:
# checks = sc.stabilizer_checks(2, 2)
print('loading qubit')
test = stim.LogicalQubit(checks=checks, num_mem=2, params=params)
print('running circuit')
circ = test.generate_stim(rounds=num_rounds)
with open("data1ms.stim", 'w') as f:
    circ.to_file(f)


dem = circ.detector_error_model(approximate_disjoint_errors=True, decompose_errors=True)


sampler = circ.compile_sampler()
one_sample = sampler.sample(shots=1)[0]
for k in range(0, len(one_sample), num_rounds+1):
    timeslice = one_sample[k:k+num_rounds+1]
    print("".join("1" if e else "_" for e in timeslice))

