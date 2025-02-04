import networkx as nx
from qiskit import QuantumCircuit
from ..objects import Logger, Qubit
from ..components import *
from .layers import *
import random
import os
import csv
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
import torch
from qiskit import QuantumCircuit

class ClassificadorQML(torch.nn.Module):
    """
    Classe para criar um modelo de aprendizado de máquina quântico (QML)
    baseado no template StronglyEntanglingLayers.
    """
    def __init__(self, dim_entrada, dim_saida, num_qubits, num_camadas):
        super().__init__()
        torch.manual_seed(1337)  # Fixar seed para reprodutibilidade
        self.num_qubits = num_qubits
        self.dim_saida = dim_saida
        self.num_camadas = num_camadas
        self.dispositivo = qml.device("lightning.qubit", wires=self.num_qubits)

        # Formato dos pesos do circuito
        self.formato_pesos = qml.StronglyEntanglingLayers.shape(
            n_layers=self.num_camadas, n_wires=self.num_qubits
        )

        # Criar o QNode
        self.qnode = qml.QNode(self.circuito, self.dispositivo)

    def circuito(self, inputs, pesos, viés):
        """Define o circuito quântico do modelo."""
        inputs = np.random.rand(*self.formato_pesos)  # Gera inputs aleatórios
        inputs = torch.tensor(inputs, dtype=torch.float32)  
        inputs = torch.reshape(inputs, self.formato_pesos)

        qml.StronglyEntanglingLayers(
            weights=pesos * inputs + viés, wires=range(self.num_qubits)
        )
        return [qml.expval(qml.PauliZ(i)) for i in range(self.dim_saida)]

    def gerar_qiskit_circuit(self):
        """
        Transforma o circuito PennyLane em um circuito Qiskit compatível com a rede.
        """
        qc = QuantumCircuit(self.num_qubits)

        # Criando pesos e viés fictícios
        pesos_falsos = np.random.rand(*self.formato_pesos)
        vies_falso = np.random.rand(*self.formato_pesos)
        entradas_falsas = np.random.rand(*self.formato_pesos)  

        # Extraindo operações
        with qml.tape.QuantumTape() as tape:
            self.qnode(entradas_falsas, pesos_falsos, vies_falso)

        for op in tape.operations:
            nome_puerta = op.name
            qubits = op.wires.tolist()

            if nome_puerta == "RX":
                qc.rx(np.random.uniform(0, 2 * np.pi), qubits[0])
            elif nome_puerta == "RY":
                qc.ry(np.random.uniform(0, 2 * np.pi), qubits[0])
            elif nome_puerta == "RZ":
                qc.rz(np.random.uniform(0, 2 * np.pi), qubits[0])
            elif nome_puerta == "CNOT":
                qc.cx(qubits[0], qubits[1])
            elif nome_puerta == "CZ":
                qc.cz(qubits[0], qubits[1])
            elif nome_puerta == "SWAP":
                qc.swap(qubits[0], qubits[1])

        return qc
