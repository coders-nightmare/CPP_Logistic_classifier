#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>

class LogisticRegression
{
private:
    std::vector<double> weights;

public:
    void train(const std::vector<std::vector<double>> &X_train, const std::vector<int> &y_train);
    double sigmoid(double z);
    double predict(const std::vector<double> &X);
    double evaluate(const std::vector<std::vector<double>> &X_test, const std::vector<int> &y_test);
};

#endif
