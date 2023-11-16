"""
Ripple adder example based on Cuccaro et al., quant-ph/0410184.

"""
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute
backend = BasicAer.get_backend('qasm_simulator')
coupling_map = [[0, 1], [0, 8], [1, 2], [1, 9], [2, 3], [2, 10], [3, 4], [3, 11], [4, 5], [4, 12], [5, 6], [5, 13], [6, 7], [6, 14], [7, 15], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]]
n = 2
a = QuantumRegister(n, 'a')
b = QuantumRegister(n, 'b')
cin = QuantumRegister(1, 'cin')
cout = QuantumRegister(1, 'cout')
ans = ClassicalRegister(n + 1, 'ans')
qc = QuantumCircuit(a, b, cin, cout, ans, name='rippleadd')

def majority(p, a, b, c):
    if False:
        print('Hello World!')
    'Majority gate.'
    p.cx(c, b)
    p.cx(c, a)
    p.ccx(a, b, c)

def unmajority(p, a, b, c):
    if False:
        i = 10
        return i + 15
    'Unmajority gate.'
    p.ccx(a, b, c)
    p.cx(c, a)
    p.cx(a, b)
adder_subcircuit = QuantumCircuit(cin, a, b, cout)
majority(adder_subcircuit, cin[0], b[0], a[0])
for j in range(n - 1):
    majority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
adder_subcircuit.cx(a[n - 1], cout[0])
for j in reversed(range(n - 1)):
    unmajority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
unmajority(adder_subcircuit, cin[0], b[0], a[0])
qc.x(a[0])
qc.x(b)
qc &= adder_subcircuit
for j in range(n):
    qc.measure(b[j], ans[j])
qc.measure(cout[0], ans[n])
job = execute(qc, backend=backend, coupling_map=None, shots=1024)
result = job.result()
print(result.get_counts(qc))
job = execute(qc, backend=backend, coupling_map=coupling_map, shots=1024)
result = job.result()
print(result.get_counts(qc))