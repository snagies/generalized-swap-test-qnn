import numpy as np

from qiskit_machine_learning.circuit.library import RawFeatureVector, QNNCircuit
from qiskit.providers.basic_provider import BasicProvider
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
#from qiskit_machine_learning.neural_networks import SamplerQNN
#from qiskit.circuit import Parameter, ParameterVector
from helpers import normalize

def swap_test(dim, fm_explicit = False):
    
    cr = ClassicalRegister(1, 'outcome')
    control = QuantumRegister(1, 'control')
    x = QuantumRegister(dim, 'x')
    weights = QuantumRegister(dim, 'weights')
    
    qc = QuantumCircuit(cr, control, x, weights)
    
    #ansatz
    qc.h(control)

    for i in range(dim):
        qc.cswap(control, x[i], weights[i])
    
    qc.h(control)
    
    qc.measure(0, 0)
    
    if fm_explicit:
        
        fm = QuantumCircuit(cr, control, x, weights)
        
        #feature map
        fm.compose(RawFeatureVector(2**dim), x, inplace = True)
        
        qc.compose(RawFeatureVector(2**dim), weights, front = True, inplace = True)
        
        return fm, qc
    
    return qc


test = normalize(np.random.rand(4))
test2 = normalize(np.random.rand(4))
test2 = test2 - test*np.dot(test,test2)

fm, ae = swap_test(2, fm_explicit = True)
fm.assign_parameters(test, inplace = True)
ae.assign_parameters(test2, inplace = True)

qc = QNNCircuit(feature_map = fm, ansatz = ae)



provider = BasicProvider()
backend = provider.get_backend("basic_simulator")



'''
trans_qc = transpile(qc, backend)
job = backend.run(trans_qc, shots = 1024)
results  = job.result()
counts = results.get_counts()
'''
