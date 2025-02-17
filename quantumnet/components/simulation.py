# import pennylane as qml
# from pennylane import numpy as np
# import torch
# from qiskit import QuantumCircuit

# # Parâmetros do classificador quântico
# input_dim = 256
# num_classes = 4
# num_layers = 32   # Profundidade do circuito
# num_qubits = 8    # Número de qubits

# class QML_classifier(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, num_qubits, num_layers):
#         super().__init__()
#         torch.manual_seed(1337)
#         self.num_qubits = num_qubits
#         self.output_dim = output_dim
#         self.num_layers = num_layers
#         self.device = qml.device("lightning.qubit", wires=self.num_qubits)
#         self.weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_qubits)

#         @qml.qnode(self.device, interface="torch")
#         def circuit(sample, weights, bias):
#             qml.StronglyEntanglingLayers(weights=weights * sample + bias, wires=range(self.num_qubits))
#             return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]
        
#         self.qnode = circuit
#         init_weights = 0.1 * torch.rand(self.weights_shape, requires_grad=True)
#         init_bias = 0.1 * torch.rand(self.weights_shape, requires_grad=True)
#         self.weights = torch.nn.Parameter(init_weights)
#         self.bias = torch.nn.Parameter(init_bias)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         expected_elements = int(np.prod(self.weights_shape))
#         if x.shape[1] > expected_elements:
#             x = x[:, :expected_elements]
#         elif x.shape[1] < expected_elements:
#             repeat_factor = (expected_elements // x.shape[1]) + 1
#             x = x.repeat(1, repeat_factor)[:, :expected_elements]
#         outputs = []
#         for i in range(batch_size):
#             sample = x[i].reshape(self.weights_shape)
#             output = self.qnode(sample, self.weights, self.bias)
#             if isinstance(output, (list, tuple)):
#                 output = torch.stack(output)
#             outputs.append(output)
#         return torch.stack(outputs)

# # Wrapper para reutilizar o QNode
# class QNodeWrapper:
#     def __init__(self, qnode, depth):
#         self.qnode = qnode
#         self._depth = depth

#     def depth(self):
#         return self._depth

# def setup_model():
#     qml_model = QML_classifier(input_dim, num_classes, num_qubits, num_layers)
#     custom_qc = QNodeWrapper(qml_model.qnode, num_layers)
#     return qml_model, custom_qc

# def train_model(qml_model, optimizer, loss_fn, X_train, Y_train, X_test, Y_test, epochs=4):
#     def accuracy(labels, predictions):
#         acc = sum(torch.argmax(p) == l for l, p in zip(labels, predictions)) / len(labels)
#         return acc

#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         outputs = qml_model(X_train)
#         batch_loss = loss_fn(outputs, Y_train)
#         batch_loss.backward()
#         optimizer.step()
#         with torch.no_grad():
#             predictions = qml_model(X_test)
#             acc = accuracy(Y_test, predictions)
#             print(f"Epoch {epoch+1}/{epochs} - Loss: {batch_loss.item():.4f}, Acurácia: {acc:.4f}")
#     print("Treinamento finalizado com sucesso!")
#     return qml_model


# class ClassificadorQML(torch.nn.Module):
#     """
#     Classe que cria um modelo de aprendizado de máquina quântico (QML)
#     baseado no template StronglyEntanglingLayers.
#     """
#     def __init__(self, dim_entrada, dim_saida, num_qubits, num_camadas):
#         super().__init__()
#         torch.manual_seed(1337)  # Fixar seed para reprodutibilidade
#         self.num_qubits = num_qubits
#         self.dim_saida = dim_saida
#         self.num_camadas = num_camadas
#         self.dispositivo = qml.device("lightning.qubit", wires=self.num_qubits)

#         # Formato dos pesos do circuito
#         self.formato_pesos = qml.StronglyEntanglingLayers.shape(
#             n_layers=self.num_camadas, n_wires=self.num_qubits
#         )

#         # Criar o QNode
#         self.qnode = qml.QNode(self.circuito, self.dispositivo)

#     def circuito(self, inputs, pesos, viés):
#         """Define o circuito quântico do modelo."""

#         # **Garantindo que tudo seja um tensor do PyTorch**
#         inputs = torch.tensor(inputs, dtype=torch.float32)  
#         pesos = torch.tensor(pesos, dtype=torch.float32)
#         viés = torch.tensor(viés, dtype=torch.float32)

#         # **Redimensionando os inputs para o formato correto**
#         inputs = torch.reshape(inputs, self.formato_pesos)

#         # **Usando os valores corretamente formatados no circuito**
#         qml.StronglyEntanglingLayers(
#             weights=pesos * inputs + viés, wires=range(self.num_qubits)
#         )
#         return [qml.expval(qml.PauliZ(i)) for i in range(self.dim_saida)]


#     def gerar_qiskit_circuit(self):
#         """
#         Transforma o circuito PennyLane em um circuito Qiskit compatível com a rede.
#         """
#         qc = QuantumCircuit(self.num_qubits)

#         # Criando pesos e viés fictícios
#         pesos= np.random.rand(*self.formato_pesos)
#         vies = np.random.rand(*self.formato_pesos)
#         entradas = np.random.rand(*self.formato_pesos)  

#         # Extraindo operações
#         with qml.tape.QuantumTape() as tape:
#             self.qnode(entradas, pesos, vies)

#         for op in tape.operations:
#             nome_puerta = op.name
#             qubits = op.wires.tolist()

#             if nome_puerta == "RX":
#                 qc.rx(np.random.uniform(0, 2 * np.pi), qubits[0])
#             elif nome_puerta == "RY":
#                 qc.ry(np.random.uniform(0, 2 * np.pi), qubits[0])
#             elif nome_puerta == "RZ":
#                 qc.rz(np.random.uniform(0, 2 * np.pi), qubits[0])
#             elif nome_puerta == "CNOT":
#                 qc.cx(qubits[0], qubits[1])
#             elif nome_puerta == "CZ":
#                 qc.cz(qubits[0], qubits[1])
#             elif nome_puerta == "SWAP":
#                 qc.swap(qubits[0], qubits[1])

#         return qc
    