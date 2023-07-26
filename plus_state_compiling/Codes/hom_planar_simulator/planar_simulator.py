from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from plus_state_compiling.Codes.surface_code import stabilizer_checks
from plus_state_compiling.Codes.reed_muller import reed_muller_checks

def create_checks_circuit(checks_list, n_data_qubits, n_syndrome_qubits):
    n_qubits = n_data_qubits + n_syndrome_qubits
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_syndrome_qubits)
    circuit = QuantumCircuit(qr, cr)

    syndrome_qubit_idx = n_data_qubits
    for check_type, check_qubits in checks_list:
        # Find the index of the syndrome qubit for this check

        if check_type == 'X':

            circuit.h(syndrome_qubit_idx)
            for qubit in check_qubits:
                circuit.cx(syndrome_qubit_idx, qubit)
            circuit.h(syndrome_qubit_idx)

        elif check_type == 'Z':

            for qubit in check_qubits:
                circuit.cx(qubit, syndrome_qubit_idx)


        else:
            raise ValueError("Invalid check_type. Must be 'X' or 'Z'.")

        syndrome_qubit_idx += 1

    # Measure all syndrome qubits simultaneously
    circuit.measure(qr[n_data_qubits:], cr)

    return circuit


if __name__ == '__main__':
    # Checks for Reed-Muller code:
    # X_checks, Z_checks = reed_muller_checks()
    # checks = X_checks + Z_checks

    # Checks for (Lx, Ly) surface code:
    checks = stabilizer_checks(3, 3)

    # max of the second element of each tuple in checks
    n_data_qubits = max([max(qubits) for _, qubits in checks]) + 1
    n_syndrome_qubits = len(checks)

    circuit = create_checks_circuit(checks, n_data_qubits, n_syndrome_qubits)

    circuit.draw(output='mpl', filename='checks_circuit.png', scale=0.5)