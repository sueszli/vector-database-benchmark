from qiskit_experiments.library import StateTomography
import qiskit

class StateTomographyBench:
    params = [2, 3, 4, 5]
    param_names = ['n_qubits']
    version = '0.3.0'
    timeout = 120.0

    def setup(self, _):
        if False:
            for i in range(10):
                print('nop')
        self.qasm_backend = qiskit.BasicAer.get_backend('qasm_simulator')

    def time_state_tomography_bell(self, n_qubits):
        if False:
            print('Hello World!')
        meas_qubits = [n_qubits - 2, n_qubits - 1]
        qr_full = qiskit.QuantumRegister(n_qubits)
        bell = qiskit.QuantumCircuit(qr_full)
        bell.h(qr_full[meas_qubits[0]])
        bell.cx(qr_full[meas_qubits[0]], qr_full[meas_qubits[1]])
        qst_exp = StateTomography(bell, measurement_qubits=meas_qubits)
        expdata = qst_exp.run(self.qasm_backend, shots=5000).block_for_results()
        expdata.analysis_results('state')
        expdata.analysis_results('state_fidelity')

    def time_state_tomography_cat(self, n_qubits):
        if False:
            while True:
                i = 10
        qr = qiskit.QuantumRegister(n_qubits, 'qr')
        circ = qiskit.QuantumCircuit(qr, name='cat')
        circ.h(qr[0])
        for i in range(1, n_qubits):
            circ.cx(qr[0], qr[i])
        qst_exp = StateTomography(circ)
        expdata = qst_exp.run(self.qasm_backend, shots=5000).block_for_results()
        expdata.analysis_results('state')
        expdata.analysis_results('state_fidelity')