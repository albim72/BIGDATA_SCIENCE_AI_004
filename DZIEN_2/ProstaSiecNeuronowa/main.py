import numpy as np
from simplenn import SimpleNeuralNetwork

network = SimpleNeuralNetwork()
print(network)

input_data = np.array([[1,1,0],[1,1,1],[1,1,0],[1,0,0],[0,1,0],[0,0,0],[0,0,1]])
output_data = np.array([[1, 0, 1, 1, 1, 1, 0]]).T
epochs = 50_000

network.train(input_data,output_data,epochs)
print(f"wytrenowany wektor wag:\n{network.weights}")

print("_________ predykcja __________")
test_data = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])

for data in test_data:
    print(f"wynik dla {data} -> {network.propagation(data)}")



