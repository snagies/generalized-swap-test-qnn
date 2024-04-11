import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_machine_learning.circuit.library import RawFeatureVector, QNNCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit import Parameter, ParameterVector


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def swap_test(dim):
    
    control = QuantumRegister(1, 'control')
    x = QuantumRegister(dim, 'x')
    weights = QuantumRegister(dim, 'weights')
    
    qc = QuantumCircuit(control, x, weights)
    fm = QuantumCircuit(control, x, weights)
    
    #feature map
    fm.compose(RawFeatureVector(2**dim), x, inplace = True)
    
    #ansatz
    qc.h(control)

    for i in range(dim):
        qc.cswap(control, x[i], weights[i])
    
    qc.h(control)
    
    qc.compose(RawFeatureVector(2**dim), weights, front = True, inplace = True)
    
    return fm, qc


test = normalize(np.random.rand(4))

fm, ae = swap_test(2)
#fm.assign_parameters(test, inplace = True)
#ae.assign_parameters(test, inplace = True)

qc = QNNCircuit(feature_map = fm, ansatz = ae)

#qnn = SamplerQNN(circuit=qc)