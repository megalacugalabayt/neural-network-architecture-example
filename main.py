import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Sigmoid
def sigmoid(x):
    clipped_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-clipped_x))

#Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.bias_hidden = np.random.uniform(-1, 1, (hidden_size, 1))
        self.weights_hidden_output = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (output_size, 1))

    def feedforward(self, inputs):
        hidden_inputs = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden
        hidden_outputs = sigmoid(hidden_inputs)
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
        final_outputs = sigmoid(final_inputs)
        return final_outputs

    def train(self, inputs, targets, validation_data, learning_rate=0.1, epochs=10000, accuracy_threshold=None):
        history_loss = []
        history_accuracy = []
        validation_loss = []
        validation_accuracy = []

        X_val, y_val = validation_data

        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0

            # Training
            for i in range(len(inputs)):

                input_data = np.array(inputs[i], ndmin=2).T
                target_data = np.array(targets[i], ndmin=2).T

                #hidden layer outputs
                hidden_inputs = np.dot(self.weights_input_hidden, input_data) + self.bias_hidden
                hidden_outputs = sigmoid(hidden_inputs)

                #final layer outputs
                final_inputs = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
                final_outputs = sigmoid(final_inputs)

                # Backpropagation
                output_errors = target_data - final_outputs
                output_gradients = final_outputs * (1 - final_outputs) * output_errors

                hidden_errors = np.dot(self.weights_hidden_output.T, output_gradients)
                hidden_gradients = hidden_outputs * (1 - hidden_outputs) * hidden_errors

                self.weights_hidden_output += learning_rate * np.dot(output_gradients, hidden_outputs.T)
                self.bias_output += learning_rate * output_gradients

                self.weights_input_hidden += learning_rate * np.dot(hidden_gradients, input_data.T)
                self.bias_hidden += learning_rate * hidden_gradients

                total_loss += np.mean(np.abs(output_errors))

                prediction = 1 if final_outputs > 0.5 else 0
                correct_predictions += (prediction == targets[i])

            average_loss = total_loss / len(inputs)
            accuracy = correct_predictions / len(inputs)

            # Validation
            val_total_loss = 0
            val_correct_predictions = 0

            for i in range(len(X_val)):
                val_input_data = np.array(X_val[i], ndmin=2).T
                val_target_data = np.array(y_val[i], ndmin=2).T

                val_hidden_inputs = np.dot(self.weights_input_hidden, val_input_data) + self.bias_hidden
                val_hidden_outputs = sigmoid(val_hidden_inputs)

                val_final_inputs = np.dot(self.weights_hidden_output, val_hidden_outputs) + self.bias_output
                val_final_outputs = sigmoid(val_final_inputs)

                val_output_errors = val_target_data - val_final_outputs
                val_total_loss += np.mean(np.abs(val_output_errors))

                val_prediction = 1 if val_final_outputs > 0.5 else 0
                val_correct_predictions += (val_prediction == y_val[i])


            average_val_loss = val_total_loss / len(X_val)
            val_accuracy = val_correct_predictions / len(X_val)

            history_loss.append(average_loss)
            history_accuracy.append(accuracy)
            validation_loss.append(average_val_loss)
            validation_accuracy.append(val_accuracy)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, \n"
                      f"Train Loss: {average_loss}, Train Acc: {accuracy}, \n"
                      f"Validation Loss: {average_val_loss}, Validation Acc: {val_accuracy}")


            if accuracy_threshold is not None and accuracy >= accuracy_threshold:
                print(f"Training stopped early at epoch {epoch} as accuracy reached {accuracy_threshold}")
                break

        final_accuracy = history_accuracy[-1]
        print(f"Final Training Accuracy: {final_accuracy}")

        return history_loss, history_accuracy, validation_loss, validation_accuracy


np.random.seed(42) 
random.seed(42)


num_samples = 1000
input_data = np.random.rand(num_samples, 3) * 100 
output_data = np.random.randint(2, size=(num_samples, 1))  


X_train, X_val, y_train, y_val = train_test_split(input_data, output_data, test_size=0.2, random_state=42)


input_size = 3
hidden_size = 4
output_size = 1
nn = NeuralNetwork(input_size, hidden_size, output_size)


accuracy_threshold = 0.95  
history_loss, history_accuracy, val_loss, val_accuracy = nn.train(X_train, y_train, validation_data=(X_val, y_val), accuracy_threshold=accuracy_threshold)


plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(history_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(history_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(3, 1, 3)

predictions = [1 if nn.feedforward(np.array(x, ndmin=2).T) > 0.5 else 0 for x in X_val]


cm = confusion_matrix(y_val, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix for Test Data')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.tight_layout()
plt.show()


