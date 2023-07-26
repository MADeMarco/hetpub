import numpy as np
import stim
import plus_state_compiling.Compiling.hetarch_USC_compiler as compile
import warnings
import pandas as pd


class LogicalQubit():
    def __init__(self, checks: list, num_mem: int, params: dict) -> None:

        self.storage_T1 = params['storage_T1']  # storage
        self.storage_T2 = params['storage_T2']
        self.compute_T1 = params['compute_T1']  # compute
        self.compute_T2 = params['compute_T2']

        self.gate2_err = params['gate2_err']
        self.time_2q = params['time_2q']

        self.readout_err = params['readout_err']
        self.time_measurement_compute = params['time_measurement_compute']

        self.reset_err = params['reset_err']
        self.time_reset_compute = params['time_reset_compute']

        self.gate1_err_compute = params['compute_1q_err']
        self.time_1q_compute = params['time_1q_compute']

        self.err_save_load = params['err_save_load']
        self.time_save_load = params['time_save_load']

        # Get number of qubits:
        flattened = [num for sublist in [lst[1] for lst in checks] for num in sublist]
        self.num_qubits = max(flattened) + 1
        # Set number of memories
        self.num_mem = num_mem

        # Partition qubits between memories:
        self.qubit_assignments = compile.get_storage_assignments(checks, self.num_qubits, num_mem)
        # Assemble a dataframe labeling where qubits are:
        self.qubit_df = compile.initialize_assign_qubits(self.qubit_assignments, self.num_qubits, self.num_mem, F=0)
        # Schedule the stabilizer checks:
        self.stabilizer_schedule = compile.schedule_checks(checks, self.qubit_assignments)

        # Program qubit groupings:
        self.ancillas = self.qubit_df.loc[self.qubit_df['Type'] == 'Ancilla', 'QbitID'].tolist()
        self.datas = self.qubit_df.loc[self.qubit_df['Type'] == 'Data', 'QbitID'].tolist()
        self.flags = self.qubit_df.loc[self.qubit_df['Type'] == 'Flag', 'QbitID'].tolist()
        self.all_qubits = self.ancillas + self.datas + self.flags

        px = (1. - np.exp(-1 * self.time_1q_compute / self.compute_T1)) / 4
        pz = (1. - np.exp(-1 * self.time_1q_compute / self.compute_T2)) / 2 - px
        # idle XYZ errors on compute qubits during a 1Q gate on a compute qubit
        self.idle_noise_on_compute_1q = [px, px, pz]

        px = (1. - np.exp(-1 * self.time_1q_compute / self.storage_T1)) / 4
        pz = (1. - np.exp(-1 * self.time_1q_compute / self.storage_T2)) / 2 - px
        # idle XYZ errors on storage qubits during a 1Q gate on a compute qubit
        self.idle_noise_on_storage_1q = [px, px, pz]

        px = (1. - np.exp(-1 * self.time_2q / self.compute_T1)) / 4
        pz = (1. - np.exp(-1 * self.time_2q / self.compute_T2)) / 2 - px
        # idle XYZ errors on compute qubits during a 2Q gate
        self.idle_noise_on_compute_2q = [px, px, pz]

        px = (1. - np.exp(-1 * self.time_2q / self.storage_T1)) / 4
        pz = (1. - np.exp(-1 * self.time_2q / self.storage_T2)) / 2 - px
        # idle XYZ errors on storage qubits during a 2Q gate
        self.idle_noise_on_storage_2q = [px, px, pz]

        px = (1. - np.exp(-1 * self.time_reset_compute / self.storage_T1)) / 4
        pz = (1. - np.exp(-1 * self.time_reset_compute / self.storage_T2)) / 2 - px
        # idle XYZ errors on storage qubits during reset on the compute qubits
        self.idle_noise_on_storage_reset = [px, px, pz]

        px = (1. - np.exp(-1 * self.time_measurement_compute / self.storage_T1)) / 4
        pz = (1. - np.exp(-1 * self.time_measurement_compute / self.storage_T2)) / 2 - px
        # idle XYZ errors on storage qubits during measurement on the syndrome qubits
        self.idle_noise_on_storage_meas = [px, px, pz]

        px = (1. - np.exp(-1 * self.time_save_load / self.storage_T1)) / 4
        pz = (1. - np.exp(-1 * self.time_save_load / self.storage_T2)) / 2 - px
        # idle XYZ errors on storage qubits during save/load
        self.idle_noise_on_storage_save = [px, px, pz]

        px = (1. - np.exp(-1 * self.time_save_load / self.compute_T1)) / 4
        pz = (1. - np.exp(-1 * self.time_save_load / self.compute_T1)) / 2 - px
        # idle XYZ errors on compute qubits during save/load
        self.idle_noise_on_compute_save = [px, px, pz]

        self.meas_record = pd.DataFrame(columns=['Round Number', 'XorZ', 'Check', 'Stabilizer Number', 'Measurement Index'])

    def qid_to_df_index(self, qid):
        index = self.qubit_df.index[self.qubit_df['QbitID'] == qid].tolist()
        if len(index) > 1:
            warnings.warn("More than one qubit with the same qid")
        return index[0]

    def filter_qubits(self, properties):
        # Create a copy of the original DataFrame
        filtered_df = self.qubit_df.copy()

        # Filter the DataFrame for each column and allowed value
        for col, allowed_values in properties.items():
            filtered_df = filtered_df[filtered_df[col].isin(allowed_values)]

        qids = filtered_df['QbitID'].tolist()
        return qids

    def apply_gate(self, circ, gate, perfect_round):

        gate_type = gate[0]
        qubits = gate[1]

        if gate_type == 'SWAP':

            qid1 = qubits[0][0][0]
            qid2 = qubits[1][0][0]

            # Swap the qubit assignments in the dataframe:
            index1 = self.qid_to_df_index(qid1)
            index2 = self.qid_to_df_index(qid2)
            status1 = self.qubit_df.loc[index1, 'Status']
            status2 = self.qubit_df.loc[index2, 'Status']
            self.qubit_df.loc[index1, 'Status'] = status2
            self.qubit_df.loc[index2, 'Status'] = status1

            if qid1 == qid2:
                print(gate)
                print("CNOT on the same qubit")

            # Apply depolarizing noise during the swap:
            if not perfect_round:
                circ.append("DEPOLARIZE2", [qid1, qid2], self.err_save_load)

        elif gate_type == 'CNOT':

            qid1 = qubits[0][0][0]
            qid2 = qubits[1][0][0]



            # Do a CNOT on the qubits:
            circ.append("CNOT", [qid1, qid2])
            # Apply depolarizing noise during the CNOT:
            if not perfect_round:
                circ.append("DEPOLARIZE2", [qid1, qid2], self.gate2_err)

        elif gate_type == "H":

            qid1 = qubits[0][0][0]

            circ.append("H", [qid1])
            if not perfect_round:
                circ.append("DEPOLARIZE1", [qid1], self.gate1_err_compute)

        else:
            warnings.warn("something is wrong with the gate gate_type")

    def apply_timestep(self, circ, gate_instruction, perfect_round):

        timestep = gate_instruction.pop(0)

        # Apply the gates in the timestep:
        for gate in gate_instruction:
            self.apply_gate(circ, gate, perfect_round)

        # Apply idle noise on the qubits. Note that we assume that all timesteps
        # take the equivalent of a 2Q gate time.
        if not perfect_round:
            compute_qubits = self.filter_qubits({'Status': ['Compute']})
            storage_qubits = self.filter_qubits({'Status': ['Storage']})
            circ.append("PAULI_CHANNEL_1", compute_qubits, self.idle_noise_on_compute_2q)
            circ.append("PAULI_CHANNEL_1", storage_qubits, self.idle_noise_on_storage_2q)



    def apply_stabilizer_step(self, circ, instruction, stabilizer_number, first=False, perfect_round=False):

        # Get the schedule for individual gates from the compiler:
        gate_schedule = compile.schedule_individual_gates(self.qubit_assignments, instruction, self.qubit_df)

        # Apply the gates in the schedule, including errors during each timestep:
        for gate_instruction in gate_schedule:
            self.apply_timestep(circ, gate_instruction, perfect_round)

        # Measure the ancilla qubits, with error (assuming we always measure all
        # ancilla at the end of a stabilizer step):

        self.measure_ancilla(circ, stabilizer_number, XorZ=instruction[0][0], check=instruction[0][1],
                             first=first, perfect_round=perfect_round)

        circ.append("TICK")

    def get_meas_rec(self, stabilizer_number, rounds_back=0):
        temp = self.meas_record[self.meas_record['Stabilizer Number'] == stabilizer_number]
        index = sorted(temp['Measurement Index'].astype(int), reverse=True)[-rounds_back]
        return index

    def measure_ancilla(self, circ, stabilizer_number, XorZ, check, first=False, perfect_round=False):
        for ancilla in self.ancillas:
            if not perfect_round:
                circ.append("X_ERROR", [ancilla], self.reset_err)

            circ.append("M", [ancilla])

            # Update the measurement record:
            """Needs to be fixed for multiple ancilla measurements!"""
            self.meas_record['Measurement Index'] -= 1  # decrement the index of all previous measurments
            new_measurement = pd.DataFrame({'Round Number': [1], 'XorZ': [XorZ], 'Check': [check], 'Stabilizer Number': [stabilizer_number],
                                            'Measurement Index': [-1]})
            self.meas_record = pd.concat([self.meas_record, new_measurement], ignore_index=True)

            # Set up the detector:
            if not first:
                # Want to compare the perfect_round two measurements of the same stabilizer:
                rec1 = stim.target_rec(self.get_meas_rec(stabilizer_number, 0))
                rec2 = stim.target_rec(self.get_meas_rec(stabilizer_number, 1))
                circ.append("DETECTOR", [rec1, rec2], stabilizer_number)
            else:
                rec = stim.target_rec(self.get_meas_rec(stabilizer_number, 0))
                circ.append("DETECTOR", [rec], stabilizer_number)

        # Apply idle noise on the other qubits:
        if not perfect_round:
            storage_qubits = self.filter_qubits({'Type': ['Data', 'Flag']})
            circ.append("PAULI_CHANNEL_1", storage_qubits, self.idle_noise_on_storage_meas)

    def measure_logical_Z(self, circ, perfect_round=False):
        instruction = ('Z', self.datas) # Only works for Reed-Muller codes
        self.apply_stabilizer_step(circ, [instruction], stabilizer_number=1.111,# 1.111 labels logical Z ops.
                                   first=True, perfect_round=perfect_round)


    def reset_qubits(self, circ, qubits, perfect_round=False):

        circ.append("R", qubits)
        if not perfect_round:
            # include measurement error:
            circ.append("X_ERROR", qubits, self.reset_err)

            # and apply idle noise on the other qubits:
            compute_qubits = self.filter_qubits({'Status': ['Compute']})
            storage_qubits = self.filter_qubits({'Status': ['Storage']})
            circ.append("PAULI_CHANNEL_1", compute_qubits, self.idle_noise_on_compute_2q)
            circ.append("PAULI_CHANNEL_1", storage_qubits, self.idle_noise_on_storage_2q)

    def syndrome_round(self, circ: stim.Circuit, first=False, perfect_round=False) -> None:
        # Reset the system - all qubits if it's the first round:
        if first:
            self.reset_qubits(circ, self.all_qubits, perfect_round=perfect_round)
        else:
            self.reset_qubits(circ, self.ancillas, perfect_round=perfect_round)

        # Apply the stabilizer steps:
        schedule = self.stabilizer_schedule

        for idx, stabilizer_step in enumerate(schedule):
            self.apply_stabilizer_step(circ, stabilizer_step, stabilizer_number=idx, first=first,
                                       perfect_round=perfect_round)

        return circ

    def generate_stim(self, rounds) -> stim.Circuit:

        # Initialize the circuit:
        circ = stim.Circuit()

        # Do the first round:
        self.syndrome_round(circ, first=True, perfect_round=False)
        # Intermediate rounds:
        if rounds > 1:
            circ.append(stim.CircuitRepeatBlock(rounds - 1, self.syndrome_round(stim.Circuit(), perfect_round=False)))

        # Do a perfect_round at the end:
        self.syndrome_round(circ, first=False, perfect_round=True)

        # Measure all Z on data qubits:
        self.measure_logical_Z(circ, perfect_round=True)
        circ.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), 0)

        print(circ.diagram())
        return circ
