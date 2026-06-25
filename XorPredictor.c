#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <limits>

double sigmoid_formula(double x) { return 1 / (1 + exp(-x)); }

#define N_INPUTS 2
#define N_HIDDEN_NODES 2
#define N_OUTPUTS 1

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
        std::cout << "Pesi caricati con successo!\n";
    }
    else {
        std::cerr << "Errore: Impossibile aprire il file dei pesi '" << filename << "'\n";
        std::cerr << "Assicurati che il file esista nella stessa directory del programma.\n";
        exit(1);
    }
}

double predict(double inputs[N_INPUTS],
    double hidden_weights[N_INPUTS][N_HIDDEN_NODES],
    double output_weights[N_HIDDEN_NODES][N_OUTPUTS],
    double hidden_layer_bias[N_HIDDEN_NODES],
    double output_layer_bias[N_OUTPUTS]) {

    double hidden_layer[N_HIDDEN_NODES];
    double output_layer[N_OUTPUTS];

    for (int j = 0; j < N_HIDDEN_NODES; j++) {
        double activation = hidden_layer_bias[j];
        for (int k = 0; k < N_INPUTS; k++) {
            activation += inputs[k] * hidden_weights[k][j];
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

    return output_layer[0];
}

int main() {
    double hidden_layer_bias[N_HIDDEN_NODES];
    double output_layer_bias[N_OUTPUTS];
    double hidden_weights[N_INPUTS][N_HIDDEN_NODES];
    double output_weights[N_HIDDEN_NODES][N_OUTPUTS];

    const char* filename = "weights.txt";

    load_weights(filename, hidden_weights, output_weights, hidden_layer_bias, output_layer_bias);

    std::cout << "\nPesi del layer nascosto:\n";
    for (int i = 0; i < N_INPUTS; i++) {
        for (int j = 0; j < N_HIDDEN_NODES; j++) {
            std::cout << hidden_weights[i][j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nBias del layer nascosto:\n";
    for (int i = 0; i < N_HIDDEN_NODES; i++) {
        std::cout << hidden_layer_bias[i] << " ";
    }

    std::cout << "\n\nPesi del layer di output:\n";
    for (int i = 0; i < N_HIDDEN_NODES; i++) {
        for (int j = 0; j < N_OUTPUTS; j++) {
            std::cout << output_weights[i][j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nBias del layer di output:\n";
    for (int i = 0; i < N_OUTPUTS; i++) {
        std::cout << output_layer_bias[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Test con i valori di training:\n";
    double test_inputs[4][2] = { {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0} };

    for (int i = 0; i < 4; i++) {
        double result = predict(test_inputs[i], hidden_weights, output_weights, hidden_layer_bias, output_layer_bias);
        std::cout << "Input: " << test_inputs[i][0] << " " << test_inputs[i][1]
            << " --> Output: " << result
            << " (Interpretazione: " << (result > 0.5 ? "1" : "0") << ")\n";
    }

    char continua = 'y';

    while (tolower(continua) == 'y') {
        double user_inputs[N_INPUTS];

        std::cout << "\nInserisci i valori di input per la predizione XOR:\n";

        std::cout << "Primo valore (0 o 1): ";
        while (!(std::cin >> user_inputs[0]) || (user_inputs[0] != 0 && user_inputs[0] != 1)) {
            std::cout << "Errore: Inserisci solo 0 o 1: ";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        std::cout << "Secondo valore (0 o 1): ";
        while (!(std::cin >> user_inputs[1]) || (user_inputs[1] != 0 && user_inputs[1] != 1)) {
            std::cout << "Errore: Inserisci solo 0 o 1: ";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        double result = predict(user_inputs, hidden_weights, output_weights, hidden_layer_bias, output_layer_bias);

        std::cout << "\nRisultato della predizione XOR:\n";
        std::cout << user_inputs[0] << " XOR " << user_inputs[1] << " = " << (result > 0.5 ? "1" : "0");
        std::cout << " (valore esatto: " << result << ")\n";

        std::cout << "\nVuoi fare un'altra predizione? (y/n): ";
        std::cin >> continua;
    }

    std::cout << "\nGrazie per aver usato la rete neurale XOR!\n";

    return 0;
}
