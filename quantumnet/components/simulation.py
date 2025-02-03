import torch
import pennylane as qml

class QMLClassifier(torch.nn.Module):
    """
    Classe para um modelo de aprendizado de máquina quântico usando o StronglyEntanglingLayers.
    """
    def __init__(self, input_dim, output_dim, num_qubits, num_layers):
        super().__init__()
        torch.manual_seed(1337)  # Seed fixa para reprodutibilidade
        self.num_qubits = num_qubits
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = qml.device("lightning.qubit", wires=self.num_qubits)
        self.weights_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.num_layers, n_wires=self.num_qubits
        )

        @qml.qnode(self.device)
        def circuit(inputs, weights, bias):
            inputs = torch.reshape(inputs, self.weights_shape)
            qml.StronglyEntanglingLayers(
                weights=weights * inputs + bias, wires=range(self.num_qubits)
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]

        param_shapes = {"weights": self.weights_shape, "bias": self.weights_shape}
        init_vals = {
            "weights": 0.1 * torch.rand(self.weights_shape),
            "bias": 0.1 * torch.rand(self.weights_shape),
        }

        # Inicializa o circuito quântico
        self.qcircuit = qml.qnn.TorchLayer(
            qnode=circuit, weight_shapes=param_shapes, init_method=init_vals
        )

    def forward(self, x, num_reup=3):
        inputs_stack = torch.hstack([x] * num_reup)
        return self.qcircuit(inputs_stack)


def test_quantum_circuit():
    """Função de teste para criar e rodar o modelo."""
    input_dim = 256
    output_dim = 4
    num_qubits = 8
    num_layers = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QMLClassifier(input_dim, output_dim, num_qubits, num_layers).to(device)
    
    sample_input = torch.rand((1, input_dim)).to(device)  # Criando uma entrada aleatória
    output = model(sample_input)
    print("Saída do modelo:", output)


if __name__ == "__main__":
    test_quantum_circuit()
