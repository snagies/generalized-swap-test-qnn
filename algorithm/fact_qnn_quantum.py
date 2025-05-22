import math
import numpy as np
import torch
from algorithm.fact_qnn import FactorizedQNNClassical
from helpers import train_model
from algorithm.helpers import normalize
from algorithm.circuit_runner import CircuitRunner
from qiskit import QuantumCircuit, QuantumRegister

def data_register(data):
    size = math.ceil(math.log2(len(data)))
    # pad data
    data = np.pad(data, (0, 2 ** size - len(data)))
    data = np.array(data, dtype=np.float64)
    data = normalize(data)
    qr = QuantumRegister(size=size)
    qc = QuantumCircuit(qr)
    qc.initialize(data)#, normalize=True)
    return qc

class FactorizedQnnQuantum():

    def __init__(self, N, k, d):
        self.N = N
        self.k = k
        self.d = d

    def train_classical(self, X_train, y_train, X_test, y_test, device, **train_args):
        self.classical_model = FactorizedQNNClassical(self.N, self.k, self.d)
        train_model(self.classical_model, X_train, y_train, X_test, y_test, device, **train_args)
        self.w = self.classical_model.w.detach().numpy()  # w_ijz
        self.alpha = self.classical_model.alpha.detach().numpy()  # alpha_i
        self.beta = self.classical_model.beta.detach().numpy()  # beta
        #data registers
        self.w_registers = [[data_register(self.w[i,j]) for j in range(self.k)] for i in range(self.N)]

    def predict_classical(self, X_test):
        with torch.no_grad():
            z = self.classical_model(X_test).squeeze()
            y_pred = torch.sign(z)
        return y_pred

    def predict_quantum(self, X_test, batch_size=32, execution='statevector', shots=1024, adaptive_shots=False):
        X_test = X_test.cpu().numpy()
        circuit_runner = CircuitRunner(execution=execution, shots=shots)
        res = []
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i : i + batch_size]
            qcs = []
            for j in range(len(X_batch)):
                qcs.extend(self.get_circuits_single(X_batch[j]))
            if adaptive_shots:
                s_lists = []
                tot_shots = shots * self.N
                # distribute shots related to alpha magnitude
                for j in range(len(X_batch)):
                    s_list = [int(tot_shots * abs(a)) for a in self.alpha]
                    # normalize shots
                    s_list = np.array(s_list) / np.sum(s_list) * tot_shots
                    s_list = np.floor(s_list).astype(int)
                    s_lists.append(s_list)
                shots_list = np.concatenate(s_lists)
                # add 1 to shots if 0
                shots_list = np.where(shots_list == 0, 1, shots_list)
                results = circuit_runner.run(qcs, measured_qubits=[0], output_bits=[0], shots_list=shots_list)
            else:
                results = circuit_runner.run(qcs, measured_qubits=[0], output_bits=[0], shots_list=None)
            for j in range(len(X_batch)):
                ps = []
                for n in range(self.N):
                    p0 = results[j * self.N + n].get('0', 0) / (results[j * self.N + n].get('0', 0) + results[j * self.N + n].get('1', 0))
                    ps.append(2 * p0 - 1)
                res.append(np.dot(self.alpha, ps) + self.beta)
        y_pred = np.sign(res).squeeze()
        return torch.tensor(y_pred)

    def get_circuits_single(self, x):
        x = np.concatenate([x, [1]]) #dummy feature
        #copy data (repeat)
        x_registers = [[data_register(x) for j in range(self.k)] for i in range(self.N)]
        qcs = []
        for i in range(self.N):
            qc = QuantumCircuit()
            # keep ancilla the first qubit
            for j in reversed(range(self.k)):
                qc.tensor(self.w_registers[i][j], inplace=True)
                qc.tensor(x_registers[i][j], inplace=True)
            qc.tensor(QuantumCircuit(1, 1), inplace=True)
            #swap test
            size = math.ceil(math.log2(self.d + 1))
            qc.h(0)
            for j in range(self.k):
                for z in range(size):
                    idx = j * 2 * size + z + 1
                    qc.cswap(0, idx, idx + size)
            qc.h(0)
            #qc.measure(0, 0)
            qcs.append(qc)
        return qcs