#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

double sigmoid_formula(double x) { return 1 / (1 + exp(-x)); }
double d_sigmoid_formula(double sigmoid_output) { return sigmoid_output * (1 - sigmoid_output); }

double clip(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

double init_weights() {
    return ((double)rand() / RAND_MAX) * 0.4 - 0.2; // range: [-0.2, 0.2]
}

void shuffle(int* array, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            std::swap(array[j], array[i]);
        }
    }
}

#define N_INPUTS 2
#define N_HIDDEN_NODES 2
#define N_OUTPUTS 1
#define N_TRAINING_SETS 4

void save_weights(const char* filename, double hidden_weights[N_INPUTS][N_HIDDEN_NODES],
    double output_weights[N_HIDDEN_NODES][N_OUTPUTS],
    double hidden_layer_bias[N_HIDDEN_NODES],
    double output_layer_bias[N_OUTPUTS]) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Hidden Weights:\n";
        for (int i = 0; i < N_INPUTS; i++) {
            for (int j = 0; j < N_HIDDEN_NODES; j++) {
                file << hidden_weights[i][j] << " ";
            }
            file << "\n";
        }
        file << "Hidden Biases:\n";
        for (int i = 0; i < N_HIDDEN_NODES; i++) {
            file << hidden_layer_bias[i] << " ";
        }
        file << "\n";
        file << "Output Weights:\n";
        for (int i = 0; i < N_HIDDEN_NODES; i++) {
            for (int j = 0; j < N_OUTPUTS; j++) {
                file << output_weights[i][j] << " ";
            }
            file << "\n";
        }
        file << "Output Biases:\n";
        for (int i = 0; i < N_OUTPUTS; i++) {
            file << output_layer_bias[i] << " ";
        }
        file << "\n";
        file.close();
    }
    else {
        std::cerr << "Unable to open file for saving weights.";
    }
}

void load_weights(const char* filename, double hidden_weights[N_INPUTS][N_HIDDEN_NODES],
    double output_weights[N_HIDDEN_NODES][N_OUTPUTS],
    double hidden_layer_bias[N_HIDDEN_NODES],
    double output_layer_bias[N_OUTPUTS]) {
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;

        std::getline(file, line);

        for (int i = 0; i < N_INPUTS; i++) {
            for (int j = 0; j < N_HIDDEN_NODES; j++) {
                file >> hidden_weights[i][j];
            }
        }

        file >> std::ws;
        std::getline(file, line);

        for (int i = 0; i < N_HIDDEN_NODES; i++) {
            file >> hidden_layer_bias[i];
        }

        file >> std::ws;
        std::getline(file, line);

        for (int i = 0; i < N_HIDDEN_NODES; i++) {
            for (int j = 0; j < N_OUTPUTS; j++) {
                file >> output_weights[i][j];
            }
        }

        file >> std::ws;
        std::getline(file, line);

        for (int i = 0; i < N_OUTPUTS; i++) {
            file >> output_layer_bias[i];
        }

        file.close();
    }
    else {
        std::cerr << "Unable to open file for loading weights.";
    }
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    const double learning_rate = 0.1;
    int number_of_epochs = 1000000000;

    double hidden_layer[N_HIDDEN_NODES];
    double output_layer[N_OUTPUTS];

    double hidden_layer_bias[N_HIDDEN_NODES];
    double output_layer_bias[N_OUTPUTS];

    double hidden_weights[N_INPUTS][N_HIDDEN_NODES];
    double output_weights[N_HIDDEN_NODES][N_OUTPUTS];

    double training_inputs[N_TRAINING_SETS][N_INPUTS] = { {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f} };
    double training_outputs[N_TRAINING_SETS][N_OUTPUTS] = { {0.0f}, {1.0f}, {1.0f}, {0.0f} };

    const char* filename = "weights.txt";

    if (std::ifstream(filename)) {
        std::cout << "Caricando i pesi dal file...\n";
        load_weights(filename, hidden_weights, output_weights, hidden_layer_bias, output_layer_bias);
    }
    else {
        std::cout << "Inizializzando nuovi pesi casuali...\n";
        for (int i = 0; i < N_INPUTS; i++) {
            for (int j = 0; j < N_HIDDEN_NODES; j++) {
                hidden_weights[i][j] = init_weights();
            }
        }

        for (int i = 0; i < N_HIDDEN_NODES; i++) {
            hidden_layer_bias[i] = init_weights();
            for (int j = 0; j < N_OUTPUTS; j++) {
                output_weights[i][j] = init_weights();
            }
        }

        for (int i = 0; i < N_OUTPUTS; i++) {
            output_layer_bias[i] = init_weights();
        }
    }

    int training_set_order[] = { 0, 1, 2, 3 };

    for (int epoch = 0; epoch < number_of_epochs; epoch++) {
        shuffle(training_set_order, N_TRAINING_SETS);

        for (int x = 0; x < N_TRAINING_SETS; x++) {
            int i = training_set_order[x];

            for (int j = 0; j < N_HIDDEN_NODES; j++) {
                double activation = hidden_layer_bias[j];
                for (int k = 0; k < N_INPUTS; k++) {
                    activation += training_inputs[i][k] * hidden_weights[k][j];
                }
                hidden_layer[j] = sigmoid_formula(activation);
            }

            for (int j = 0; j < N_OUTPUTS; j++) {
                double activation = output_layer_bias[j];
                for (int k = 0; k < N_HIDDEN_NODES; k++) {
                    activation += hidden_layer[k] * output_weights[k][j];
                }
                output_layer[j] = sigmoid_formula(activation);
            }

            if (epoch % 10 == 0 && x == 0) {
                std::cout << "Epoch " << epoch << " - Input: " << training_inputs[i][0] << " " << training_inputs[i][1]
                    << " Output: " << output_layer[0]
                    << " Expected Output: " << training_outputs[i][0] << "\n";
            }

            double delta_output[N_OUTPUTS];
            for (int j = 0; j < N_OUTPUTS; j++) {
                double error = training_outputs[i][j] - output_layer[j];
                delta_output[j] = clip(error * d_sigmoid_formula(output_layer[j]), -1.0, 1.0);
            }

            double delta_hidden[N_HIDDEN_NODES];
            for (int j = 0; j < N_HIDDEN_NODES; j++) {
                double error = 0.0f;
                for (int k = 0; k < N_OUTPUTS; k++) {
                    error += delta_output[k] * output_weights[j][k];
                }
                delta_hidden[j] = clip(error * d_sigmoid_formula(hidden_layer[j]), -1.0, 1.0);
            }

            for (int j = 0; j < N_OUTPUTS; j++) {
                output_layer_bias[j] += delta_output[j] * learning_rate;
                for (int k = 0; k < N_HIDDEN_NODES; k++) {
                    output_weights[k][j] += hidden_layer[k] * delta_output[j] * learning_rate;
                }
            }

            for (int j = 0; j < N_HIDDEN_NODES; j++) {
                hidden_layer_bias[j] += delta_hidden[j] * learning_rate;
                for (int k = 0; k < N_INPUTS; k++) {
                    hidden_weights[k][j] += training_inputs[i][k] * delta_hidden[j] * learning_rate;
                }
            }
        }

        if (epoch % 10 == 0 || epoch == number_of_epochs - 1) {
            save_weights(filename, hidden_weights, output_weights, hidden_layer_bias, output_layer_bias);
        }
    }

    std::cout << "\nTest finale dopo l'addestramento:\n";
    for (int i = 0; i < N_TRAINING_SETS; i++) {
        for (int j = 0; j < N_HIDDEN_NODES; j++) {
            double activation = hidden_layer_bias[j];
            for (int k = 0; k < N_INPUTS; k++) {
                activation += training_inputs[i][k] * hidden_weights[k][j];
            }
            hidden_layer[j] = sigmoid_formula(activation);
        }

        for (int j = 0; j < N_OUTPUTS; j++) {
            double activation = output_layer_bias[j];
            for (int k = 0; k < N_HIDDEN_NODES; k++) {
                activation += hidden_layer[k] * output_weights[k][j];
            }
            output_layer[j] = sigmoid_formula(activation);
        }

        std::cout << "Input: " << training_inputs[i][0] << " " << training_inputs[i][1]
            << " Output: " << output_layer[0]
            << " Expected Output: " << training_outputs[i][0] << "\n";
    }

    std::cout << "\nFinal hidden weights:\n";
    std::cout << "[ ";
    for (int j = 0; j < N_HIDDEN_NODES; j++) {
        std::cout << "[ ";
        for (int k = 0; k < N_INPUTS; k++) {
            std::cout << hidden_weights[k][j];
            if (k < N_INPUTS - 1) std::cout << " ";
        }
        std::cout << "] ";
    }
    std::cout << "]\n";

    std::cout << "Final hidden biases:\n";
    std::cout << "[ ";
    for (int j = 0; j < N_HIDDEN_NODES; j++) {
        std::cout << hidden_layer_bias[j];
        if (j < N_HIDDEN_NODES - 1) std::cout << " ";
    }
    std::cout << "]\n";

    std::cout << "Final output weights:\n";
    std::cout << "[ ";
    for (int j = 0; j < N_OUTPUTS; j++) {
        std::cout << "[ ";
        for (int k = 0; k < N_HIDDEN_NODES; k++) {
            std::cout << output_weights[k][j];
            if (k < N_HIDDEN_NODES - 1) std::cout << " ";
        }
        std::cout << "] ";
    }
    std::cout << "]\n";

    std::cout << "Final output biases:\n";
    std::cout << "[ ";
    for (int j = 0; j < N_OUTPUTS; j++) {
        std::cout << output_layer_bias[j];
        if (j < N_OUTPUTS - 1) std::cout << " ";
    }
    std::cout << "]\n";

    return 0;
}
