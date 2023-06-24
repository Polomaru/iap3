#include <iostream>
#include <fstream>
#include <vector>
#include "Eigen/Dense"
#include <cmath>
#include <algorithm>
#include <iomanip>

/*Se utiliza la biblioteca Eigen de una manera porque da una implementación fácil de usar para realizar operaciones matriciales en C++.*/

using namespace Eigen;

struct Dataset {
    MatrixXd features;
    VectorXd labels;
};

Dataset loadDataset(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;
    std::vector<double> labels;

    if (file) {
        std::string line;
        while (std::getline(file, line)) {
            std::vector<double> rowData;
            std::istringstream iss(line);
            std::string token;
            while (std::getline(iss, token, ',')) {
                rowData.push_back(std::stod(token));
            }
            data.push_back(rowData);
            labels.push_back(rowData.back());
        }
    } else {
        std::cerr << "Error al abrir el archivo: " << filename << std::endl;
        exit(1);
    }

    int numRows = static_cast<int>(data.size());
    int numCols = static_cast<int>(data[0].size() - 1);
    MatrixXd features(numRows, numCols);
    VectorXd labelVector(numRows);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            features(i, j) = data[i][j];
        }
        labelVector(i) = labels[i];
    }

    return { features, labelVector };
}

void normalizeDataset(Dataset& dataset) {
    dataset.features = (dataset.features.rowwise() - dataset.features.colwise().mean()).array()
                       / dataset.features.colwise().norm().maxCoeff();
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double tanhActivation(double x) {
    return std::tanh(x);
}

double relu(double x) {
    return std::max(0.0, x);
}

struct ExperimentResult {
    int numLayers;
    int numNeurons;
    std::string activationFunction;
    double precision;
    double recall;
    double f1Score;
};

ExperimentResult runExperiment(const Dataset& trainDataset, const Dataset& testDataset, int numLayers, int numNeurons, const std::string& activationFunction) {
    int numFeatures = trainDataset.features.cols();
    int numClasses = trainDataset.labels.maxCoeff() + 1;

    std::vector<int> layerSizes(numLayers, numNeurons);
    layerSizes.insert(layerSizes.begin(), numFeatures);
    layerSizes.push_back(numClasses);
    std::vector<MatrixXd> weights(numLayers + 1);
    std::vector<VectorXd> biases(numLayers + 1);

    for (int i = 0; i <= numLayers; ++i) {
        int inputSize = layerSizes[i];
        int outputSize = layerSizes[i + 1];
        weights[i] = MatrixXd::Random(outputSize, inputSize);
        biases[i] = VectorXd::Zero(outputSize);
    }

    for (int epoch = 0; epoch < 100; ++epoch) {
        for (int i = 0; i < trainDataset.features.rows(); ++i) {
            VectorXd input = trainDataset.features.row(i);
            VectorXd label = VectorXd::Zero(numClasses);
            label(static_cast<int>(trainDataset.labels(i))) = 1.0;

            std::vector<VectorXd> activations(numLayers + 1);
            activations[0] = input;

            for (int j = 0; j < numLayers; ++j) {
                VectorXd hidden = (weights[j] * activations[j] + biases[j]).unaryExpr(activationFunction);
                activations[j + 1] = hidden;
            }

            VectorXd output = (weights[numLayers] * activations[numLayers] + biases[numLayers]).unaryExpr(activationFunction);

            VectorXd error = output - label;
            VectorXd delta = error;

            for (int j = numLayers; j >= 0; --j) {
                MatrixXd weightTranspose = weights[j].transpose();
                delta = weightTranspose * delta.array() * (activations[j].array() * (1 - activations[j].array()));

                MatrixXd weightUpdate = delta * activations[j].transpose();
                weights[j] -= 0.1 * weightUpdate;
                biases[j] -= 0.1 * delta;
            }
        }
    }

    int numSamples = testDataset.features.rows();
    MatrixXd predictions(numSamples, numClasses);

    for (int i = 0; i < numSamples; ++i) {
        VectorXd input = testDataset.features.row(i);
        VectorXd output = input;

        for (int j = 0; j <= numLayers; ++j) {
            output = (weights[j] * output + biases[j]).unaryExpr(activationFunction);
        }

        predictions.row(i) = output.transpose();
    }

    MatrixXd::Index maxIndex;
    predictions.maxCoeff(&maxIndex);
    VectorXd predictedLabels = maxIndex * VectorXd::Ones(numSamples);
    VectorXd trueLabels = testDataset.labels;

    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;

    for (int i = 0; i < numSamples; ++i) {
        if (predictedLabels(i) == trueLabels(i)) {
            truePositives++;
        } else {
            falsePositives++;
            falseNegatives++;
        }
    }

    double precision = static_cast<double>(truePositives) / (truePositives + falsePositives);
    double recall = static_cast<double>(truePositives) / (truePositives + falseNegatives);
    double f1Score = 2.0 * (precision * recall) / (precision + recall);

    return { numLayers, numNeurons, activationFunction, precision, recall, f1Score };
}

int main() {
    std::cout << "Using Eigen for matrix operations." << std::endl;

    Dataset trainDataset = loadDataset("train.csv");
    Dataset testDataset = loadDataset("test.csv");

    normalizeDataset(trainDataset);
    normalizeDataset(testDataset);

    std::vector<ExperimentResult> results;

    std::vector<int> numLayersValues = { 1, 2, 3 };
    std::vector<int> numNeuronsValues = { 50, 100, 200 };
    std::vector<std::string> activationFunctions = { "sigmoid", "tanh", "relu" };

    for (int numLayers : numLayersValues) {
        for (int numNeurons : numNeuronsValues) {
            for (const std::string& activationFunction : activationFunctions) {
                ExperimentResult result = runExperiment(trainDataset, testDataset, numLayers, numNeurons, activationFunction);
                results.push_back(result);
            }
        }
    }
}
