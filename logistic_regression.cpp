#include "logistic_regression.h"
#include <cmath>
#include <iostream>

#include <cmath> // for exp()

void LogisticRegression::train(const std::vector<std::vector<double>> &X_train, const std::vector<int> &y_train)
{
    // Hyperparameters
    double learning_rate = 0.1;
    int num_iterations = 1000;
    double lambda = 0.1; // Regularization parameter

    // Initialize weights
    weights.resize(X_train[0].size() + 1);
    for (int i = 0; i < weights.size(); ++i)
    {
        weights[i] = 0.0;
    }

    // Gradient descent
    for (int iter = 0; iter < num_iterations; ++iter)
    {
        std::vector<double> gradient(X_train[0].size() + 1, 0.0);
        for (int i = 0; i < X_train.size(); ++i)
        {
            // Compute prediction
            double z = weights[0];
            for (int j = 0; j < X_train[i].size(); ++j)
            {
                z += weights[j + 1] * X_train[i][j];
            }
            double pred = sigmoid(z);

            // Compute error
            double error = pred - y_train[i];

            // Update gradient
            gradient[0] += error;
            for (int j = 0; j < X_train[i].size(); ++j)
            {
                gradient[j + 1] += error * X_train[i][j];
            }
        }

        // Update weights
        for (int j = 0; j < weights.size(); ++j)
        {
            if (j == 0)
            {
                weights[j] -= learning_rate * gradient[j] / X_train.size();
            }
            else
            {
                weights[j] -= (learning_rate * gradient[j] / X_train.size()) + (lambda * weights[j] / X_train.size());
            }
        }
    }
}

double LogisticRegression::sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

double LogisticRegression::predict(const std::vector<double> &X)
{
    double z = weights[0];
    for (int i = 0; i < X.size(); ++i)
    {
        z += weights[i + 1] * X[i];
    }
    return sigmoid(z);
}

double LogisticRegression::evaluate(const std::vector<std::vector<double>> &X_test, const std::vector<int> &y_test)
{
    double correct = 0;
    for (int i = 0; i < X_test.size(); ++i)
    {
        double pred = predict(X_test[i]);
        if ((pred >= 0.5 && y_test[i] == 1) || (pred < 0.5 && y_test[i] == 0))
        {
            correct++;
        }
    }
    return correct / X_test.size();
}
