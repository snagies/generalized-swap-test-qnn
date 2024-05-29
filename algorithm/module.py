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


#%%

#test = normalize(np.random.rand(4))
#test2 = normalize(np.random.rand(4))

#test2 = normalize(test2 - test*np.dot(test,test2))
#test2 = test


#qc = swap_test(2, test, test2)


#fm, ae = swap_test(2, fm_explicit = True)
#fm.assign_parameters(test, inplace = True)
#ae.assign_parameters(test2, inplace = True)

#qc = QNNCircuit(feature_map = fm, ansatz = ae)

#qc.barrier()
#qc.draw('mpl')
#qc.measure_all()
#print(qc.num_clbits)
#%%
#from qiskit import transpile
#from qiskit_ibm_runtime import QiskitRuntimeService, Options, SamplerV2
#from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
#from qiskit_aer.primitives import Sampler
#from qiskit.primitives import Sampler

#QiskitRuntimeService.save_account(channel="ibm_quantum", token="c990be9c320e9cb9eac1d6139eebdade74c3c097ccb0ba145b3fb88759eacb6a25dfd5e2f8945c14d34d168e89cada04b2913b1b159f61b39ba6e7914a6c9989", set_as_default=True)
#service = QiskitRuntimeService()
#backend = service.get_backend("ibm_brisbane")
#backend = "ibmq_qasm_simulator"


'''
options = Options()
options.simulator.seed_simulator = 42
options.execution.shots = 1000
options.optimization_level = 0 # no optimization
options.resilience_level = 0 # no error mitigation
'''
#qc_transpiled = transpile(qc,backend=backend, optimization_level=1)

#sampler = SamplerV2(backend=backend)
#sampler.options.dynamical_decoupling.enable = True
#sampler.options.dynamical_decoupling.sequence_type = 'XY4'

#exact_sampler = Sampler()


#job = sampler.run([qc])
#job = exact_sampler.run(qc,shots= None)

#print(f"job id: {job.job_id()}")

#result = job.result()
#print(result)
#print('P(0) = ',result.quasi_dists[0][0])

