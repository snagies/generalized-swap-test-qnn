from qiskit_aer import AerSimulator
import numpy as np
from qiskit_aer.quantum_info import AerStatevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime.fake_provider import FakeTorino

class CircuitRunner():
    def __init__(self, execution='statevector', shots=1024, seed=123, **cl_args):
        self.execution = execution
        self.shots = shots
        self.seed = seed
        if self.execution == 'simulator':
            self.backend = AerSimulator()
        elif self.execution == 'statevector':
            self.backend = None
        elif self.execution == 'fake':
            self.backend = FakeTorino()
        elif self.execution == 'real':
            self.backend = QiskitRuntimeService(channel="ibm_cloud").backend('ibm_torino')
        else:
            raise ValueError('execution not recognized')

    def get_isa_qcs(self, qcs, optimization_level=2):
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=optimization_level)
        isa_qcs = pm.run(qcs)
        return isa_qcs

    def run(self, qcs, measured_qubits, output_bits, shots_list=None, optimization_level=2, classical_register_name='c'):
        results = []
        if self.execution == 'statevector':
            for qc in qcs:
                # get positive indexes
                measured_qubits_pos = [q if q >= 0 else qc.num_qubits + q for q in measured_qubits]
                output_bits_pos = [b if b >= 0 else qc.num_clbits + b for b in output_bits]
                # dict contains measured keys only
                result = AerStatevector(qc).probabilities_dict([measured_qubits_pos [i] for i in np.argsort(output_bits_pos)])
                results.append(result)
        else:
            # measure
            for qc in qcs:
                qc.measure(measured_qubits, output_bits)
            isa_qcs = self.get_isa_qcs(qcs, optimization_level)
            if not (shots_list is None):
                # set shots
                isa_qcs = [(isa_qcs[i], None, shots_list[i]) for i in range(len(isa_qcs))]
            sampler = Sampler(mode=self.backend)
            result = sampler.run(isa_qcs, shots=self.shots).result()
            for r in result:
                counts = r.data[classical_register_name].get_counts()
                shots = r.metadata['shots']
                # normalize
                results.append({k: v / shots for k, v in counts.items()})
        return results
