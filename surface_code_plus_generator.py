import numpy as np
import stim
#import sinter

class DataQubit():

    def __init__(self, name, coords) -> None:
        self.name = name
        self.coords = coords

    def __repr__(self) -> str:
        return f'{self.name}, Coords: {self.coords}'

class MeasureQubit():

    def __init__(self, name, coords, data_qubits, basis) -> None:
        self.name = name
        self.coords = coords
        self.data_qubits = data_qubits
        self.basis = basis

    def __repr__(self):
        return f'|{self.name}, Coords: {self.coords}, Basis: {self.basis}, Data Qubits: {self.data_qubits}|'

class LogicalQubit():
    def __init__(self, d: int, gate2_err:float, params:dict) -> None:
        self.d = d
        self.gate2_err = gate2_err
        self.err_multipler = gate2_err/params['compute_compute_2q_err']
        self.readout_err = params['readout_err'] * self.err_multipler
        self.gate1_err_compute = params['compute_1q_err'] * self.err_multipler
        self.reset_err = params['reset_err'] * self.err_multipler
        self.storage_T1 = params['storage_T1'] # storage
        self.storage_T2 = params['storage_T2']
        self.compute_T1 = params['compute_T1'] # compute
        self.compute_T2 = params['compute_T2']
        self.time_measurement_compute = params['time_measurement_compute']
        self.time_reset_compute = params['time_reset_compute']
        self.time_1q_compute = params['time_1q_compute']
        self.time_2q = params['time_2q']
        self.time_save_load = params['time_save_load']
        self.err_save_load = params['err_save_load'] * self.err_multipler
        self.qubit_locations = [0]*(d**2) #keep track of where the data qubits are stored, >=0: storage, <0: compute
        self.physical_ancilla = [d*d]# d*d is the qubit index of the only ancilla qubit

        px = (1.-np.exp(-1*self.time_1q_compute/self.compute_T1))/4
        pz = (1.-np.exp(-1*self.time_1q_compute/self.compute_T2))/2 - px
        #idle XYZ errors on compute qubits during a 1Q gate on a compute qubit
        self.idle_noise_on_compute_1q = [px,px,pz]

        px = (1.-np.exp(-1*self.time_1q_compute/self.storage_T1))/4
        pz = (1.-np.exp(-1*self.time_1q_compute/self.storage_T2))/2 - px
        #idle XYZ errors on storage qubits during a 1Q gate on a compute qubit
        self.idle_noise_on_storage_1q = [px,px,pz]

        px = (1.-np.exp(-1*self.time_2q/self.compute_T1))/4
        pz = (1.-np.exp(-1*self.time_2q/self.compute_T2))/2 - px
        #idle XYZ errors on compute qubits during a 2Q gate
        self.idle_noise_on_compute_2q = [px,px,pz]

        px = (1.-np.exp(-1*self.time_2q/self.storage_T1))/4
        pz = (1.-np.exp(-1*self.time_2q/self.storage_T2))/2 - px
        #idle XYZ errors on storage qubits during a 2Q gate
        self.idle_noise_on_storage_2q = [px,px,pz]

        px = (1.-np.exp(-1*self.time_reset_compute/self.storage_T1))/4
        pz = (1.-np.exp(-1*self.time_reset_compute/self.storage_T2))/2 - px
        #idle XYZ errors on storage qubits during reset on the compute qubits
        self.idle_noise_on_storage_reset = [px,px,pz]

        px = (1.-np.exp(-1*self.time_measurement_compute/self.storage_T1))/4
        pz = (1.-np.exp(-1*self.time_measurement_ancilla/self.storage_T2))/2 - px
        #idle XYZ errors on storage qubits during measurement on the syndrome qubits
        self.idle_noise_on_storage_meas = [px,px,pz]

        px = (1.-np.exp(-1*self.time_save_load/self.storage_T1))/4
        pz = (1.-np.exp(-1*self.time_save_load/self.storage_T2))/2 - px
        #idle XYZ errors on storage qubits during save/load
        self.idle_noise_on_storage_save = [px,px,pz]

        px = (1.-np.exp(-1*self.time_save_load/self.compute_T1))/4
        pz = (1.-np.exp(-1*self.time_save_load/self.compute_T1))/2 - px
        #idle XYZ errors on compute qubits during save/load
        self.idle_noise_on_compute_save = [px,px,pz]

        # data qubits are given indices that start from 0
        self.data = [DataQubit((d*x + y), (2*x, 2*y)) for x in range(d) for y in range(d)]
        data_matching = [[None for _ in range(2*d)] for _ in range(2*d)]
        
        for data_q in self.data:
            data_matching[data_q.coords[0]][data_q.coords[1]] = data_q

        q = d*d # starting from qname=d**2 we have ancilla qubits
        self.x_ancilla = []
        self.z_ancilla = []
        for x in range(-1, d):
            for y in range(-1, d):
                if (x + y) % 2 == 1 and x != -1 and x != d - 1:# is X syndrome
                    coords = (2*x + 1, 2*y + 1)
                    data_qubits = []
                    if y != d - 1: # not on right edge
                        data_qubits += [data_matching[coords[0] + 1][coords[1] + 1], data_matching[coords[0] - 1][coords[1] + 1]]
                    if y != -1: # not on left edge
                        data_qubits += [data_matching[coords[0] + 1][coords[1] - 1], data_matching[coords[0] - 1][coords[1] - 1]]
                    measure_q = MeasureQubit(q, coords, data_qubits, "X")
                    self.x_ancilla.append(measure_q)
                elif (x + y) % 2 == 0 and y != -1 and y != d - 1:# is Z syndrome
                    coords = (2*x + 1, 2*y + 1)
                    data_qubits = []
                    if x != d - 1: # not on lower edge
                        data_qubits += [data_matching[coords[0] + 1][coords[1] + 1], data_matching[coords[0] + 1][coords[1] - 1]]
                    if x != -1: # not on upper edge
                        data_qubits += [data_matching[coords[0] - 1][coords[1] + 1], data_matching[coords[0] - 1][coords[1] - 1]]
                    measure_q = MeasureQubit(q, coords, data_qubits, "Z")
                    self.z_ancilla.append(measure_q)
                q += 1

        self.all_qubits = ([data.name for data in self.data] + self.physical_ancilla) 
        self.all_qubits.sort()

        self.observable = []
        for x in range(d):
            self.observable.append(data_matching[2*x][0])
        self.meas_record = []

    def qubits_in_storage_compute(self,exclude=[]):
        #returns 2 arrays - logical qubits in storage and in compute
        #exclude: an array of qnames to exclude
        qubits_in_storage = []
        qubits_in_compute = []
        for qname in range(self.qubit_locations):#data qubits
            if qname in exclude:
                continue
            if self.qubit_locations[qname] >= 0:
                qubits_in_storage.append(qname)
            else:
                qubits_in_compute.append(qname)
        for qname in self.physical_ancilla:
            if qname in exclude:
                continue
            qubits_in_compute.append(qname)
        return qubits_in_storage, qubits_in_compute

    def apply_1gate(self, circ, gate, qubits):#assuming that 1Q gates are only applied on X ancilla qubits
        circ.append(gate, qubits)
        circ.append("DEPOLARIZE1", qubits, self.gate1_err_compute)
        in_storage, in_compute = self.qubits_in_storage_compute()
        circ.append("PAULI_CHANNEL_1", in_compute, self.idle_noise_on_compute_1q)
        circ.append("PAULI_CHANNEL_1", in_storage, self.idle_noise_on_storage_1q)
        circ.append("TICK")

    def apply_2gate(self, circ, gate, qubits):
        circ.append(gate, qubits)
        circ.append("DEPOLARIZE2", qubits, self.gate2_err)
        #apply decoherence errors on all qubits
        in_storage, in_compute = self.qubits_in_storage_compute()
        circ.append("PAULI_CHANNEL_1", in_compute, self.idle_noise_on_compute_2q)
        circ.append("PAULI_CHANNEL_1", in_storage, self.idle_noise_on_storage_2q)
        circ.append("TICK")

    def reset_meas_qubits(self, circ, op, qubits, last=False):
        if op == "R":#assume that R is either on all ancilla qubits or on all qubits
            circ.append(op, qubits)
            if len(qubits) < len(self.all_qubits):#reset on all ancilla qubits
                circ.append("X_ERROR", qubits, self.reset_err)
                circ.append("PAULI_CHANNEL_1", [data.name for data in self.data], self.idle_noise_on_data_reset)
        if op == "M":
            if not last:
                circ.append("X_ERROR", qubits, self.readout_err)
            circ.append(op, qubits)
            # Update measurement record indices
            meas_round = {}
            for i in range(len(qubits)):
                q = qubits[-(i + 1)]
                meas_round[q] = -(i + 1)
            for round in self.meas_record:
                for q, idx in round.items():
                    round[q] = idx - len(qubits)
            self.meas_record.append(meas_round)
            # add errors on idle qubits
            if not last and len(qubits) < len(self.all_qubits):#assume data qubits are idle
                circ.append("PAULI_CHANNEL_1", [data.name for data in self.data], self.idle_noise_on_data_meas)


    def get_meas_rec(self, round_idx, qubit_name):
        return stim.target_rec(self.meas_record[round_idx][qubit_name])

    def syndrome_round(self, circ: stim.Circuit, first=False) -> None:
        all_syn = ([measure.name for measure in self.x_ancilla]+
                  [measure.name for measure in self.z_ancilla])
        if first:
            self.reset_meas_qubits(circ, "R", self.all_qubits)
            #circ.append("X",[data.name for data in self.data]) # added only for initializing all data qubits in 1
        else:
            self.reset_meas_qubits(circ, "R", all_syn)
        circ.append("TICK")
        self.apply_1gate(circ, "H", [measure.name for measure in self.x_ancilla])

        for i in range(4):
            err_qubits = []
            for measure_x in self.x_ancilla:
                if measure_x.data_qubits[i] != None:
                    err_qubits += [measure_x.name, measure_x.data_qubits[i].name]
            for measure_z in self.z_ancilla:
                if measure_z.data_qubits[i] != None:
                    err_qubits += [measure_z.data_qubits[i].name, measure_z.name]
            self.apply_2gate(circ,"CX",err_qubits)

        self.apply_1gate(circ, "H", [measure.name for measure in self.x_ancilla])

        self.reset_meas_qubits(circ, "M", all_syn)

        if not first:
            circ.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])
            for ancilla in self.x_ancilla + self.z_ancilla:
                circ.append("DETECTOR", [self.get_meas_rec(-1, ancilla.name), self.get_meas_rec(-2, ancilla.name)], ancilla.coords + (0,))
        else:
            for ancilla in self.z_ancilla:
                circ.append("DETECTOR", self.get_meas_rec(-1, ancilla.name), ancilla.coords + (0,))
        circ.append("TICK")
        return circ

    def generate_stim(self, rounds) -> stim.Circuit:
        all_data = [data.name for data in self.data]
        circ = stim.Circuit()

        # Coords
        """
        for data in self.data:
            circ.append("QUBIT_COORDS", data.name, data.coords)
        for x_ancilla in self.x_ancilla:
            circ.append("QUBIT_COORDS", x_ancilla.name, x_ancilla.coords)
        for z_ancilla in self.z_ancilla:
            circ.append("QUBIT_COORDS", z_ancilla.name, z_ancilla.coords)
        """

        self.syndrome_round(circ, first=True)
        circ.append(stim.CircuitRepeatBlock(rounds - 1, self.syndrome_round(stim.Circuit())))

        self.reset_meas_qubits(circ, "M", all_data,last=True)
        circ.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        for ancilla in self.z_ancilla:
            circ.append("DETECTOR", [self.get_meas_rec(-1, data.name) for data in ancilla.data_qubits if data is not None] +\
                        [self.get_meas_rec(-2, ancilla.name)], ancilla.coords + (0,))

        circ.append("OBSERVABLE_INCLUDE", [self.get_meas_rec(-1, data.name) for data in self.observable], 0)

        return circ