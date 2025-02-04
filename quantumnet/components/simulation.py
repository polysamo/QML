import pennylane as qml
from pennylane import numpy as np
import torch
from qiskit import QuantumCircuit

class ClassificadorQML(torch.nn.Module):
    """
    Classe que cria um modelo de aprendizado de máquina quântico (QML)
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

        # **Garantindo que tudo seja um tensor do PyTorch**
        inputs = torch.tensor(inputs, dtype=torch.float32)  
        pesos = torch.tensor(pesos, dtype=torch.float32)
        viés = torch.tensor(viés, dtype=torch.float32)

        # **Redimensionando os inputs para o formato correto**
        inputs = torch.reshape(inputs, self.formato_pesos)

        # **Usando os valores corretamente formatados no circuito**
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
        pesos= np.random.rand(*self.formato_pesos)
        vies = np.random.rand(*self.formato_pesos)
        entradas = np.random.rand(*self.formato_pesos)  

        # Extraindo operações
        with qml.tape.QuantumTape() as tape:
            self.qnode(entradas, pesos, vies)

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
    