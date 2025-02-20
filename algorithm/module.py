import numpy as np

from qiskit_machine_learning.circuit.library import RawFeatureVector, QNNCircuit
from qiskit.providers.basic_provider import BasicProvider
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from algorithm.helpers import normalize

def swap_test_fm(dim, fm_explicit = False):
    
    cr = ClassicalRegister(1, 'outcome')
    control = QuantumRegister(1, 'control')
    x = QuantumRegister(dim, 'x')
    weights = QuantumRegister(dim, 'weights')
    
    qc = QuantumCircuit(cr, control, x, weights)
    qc.barrier()
    
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
        fm.barrier()
        
        qc.compose(RawFeatureVector(2**dim), weights, front = True, inplace = True)
        
        return fm, qc
    
    return qc

def swap_test(qubit_num, inputvec, weightsvec):
    
    assert 2**qubit_num == len(inputvec) and 2**qubit_num == len(weightsvec)
    
    cr = ClassicalRegister(1, 'outcome')
    control = QuantumRegister(1, 'control')
    x = QuantumRegister(qubit_num, 'x')
    weights = QuantumRegister(qubit_num, 'weights')
    
    qc = QuantumCircuit(cr, control, x, weights)
    
    qc.initialize(inputvec, x)
    qc.initialize(weightsvec, weights)
   
    qc.barrier()
    
    qc.h(control)
    
    for i in range(qubit_num):
        qc.cswap(control, x[i], weights[i])
    
    qc.h(control)
    
    qc.measure(0, 0)
    
    return qc
