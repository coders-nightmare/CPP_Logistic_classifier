#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "iris_dataset.h"
#include <numeric>

IrisDataset::IrisDataset(const std::string &filename) : filename(filename) {}

void IrisDataset::load()
{
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::vector<double> row;
        double val;
        char comma;
        while (ss >> val)
        {
            row.push_back(val);
            ss >> comma;
        }
        if (row.size() > 1)
        {
            X.push_back({row.begin(), row.end() - 1});
            std::string target = line.substr(line.find_last_of(',') + 1);
            if (target == "Iris-setosa")
            {
                y.push_back(0);
            }
            else if (target == "Iris-versicolor")
            {
                y.push_back(1);
            }
            else if (target == "Iris-virginica")
            {
                y.push_back(2);
            }
        }
    }
    file.close();
}

void IrisDataset::preprocess()
{
    // Shuffle data
    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());
    std::vector<std::vector<double>> newX(X.size());
    std::vector<int> newY(y.size());
    for (int i = 0; i < X.size(); ++i)
    {
        newX[i] = X[indices[i]];
        newY[i] = y[indices[i]];
    }
    X = newX;
    y = newY;

    // Normalize data
    for (auto &row : X)
    {
        double sum = 0.0;
        for (auto &val : row)
        {
            sum += val;
        }
        for (auto &val : row)
        {
            val /= sum;
        }
    }
}

std::tuple<std::vector<std::vector<double>>, std::vector<int>,
           std::vector<std::vector<double>>, std::vector<int>>
IrisDataset::trainTestSplit(double testSize)
{
    int splitIndex = X.size() * (1 - testSize);
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + splitIndex);
    std::vector<int> y_train(y.begin(), y.begin() + splitIndex);
    std::vector<std::vector<double>> X_test(X.begin() + splitIndex, X.end());
    std::vector<int> y_test(y.begin() + splitIndex, y.end());
    return std::make_tuple(X_train, y_train, X_test, y_test);
}

std::vector<std::vector<double>> IrisDataset::getXTrain()
{
    // Return X_train
    return X;
}

std::vector<int> IrisDataset::getYTrain()
{
    // Return y_train
    return y;
}

std::vector<std::vector<double>> IrisDataset::getXTest()
{
    // Return X_test
    return X;
}

std::vector<int> IrisDataset::getYTest()
{
    // Return y_test
    return y;
}
