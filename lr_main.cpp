#include <iostream>
#include "iris_dataset.h"
#include "logistic_regression.h"

int main()
{
    IrisDataset dataset("iris.csv");
    dataset.load();
    dataset.preprocess();

    LogisticRegression model;
    model.train(dataset.getXTrain(), dataset.getYTrain());

    double accuracy = model.evaluate(dataset.getXTest(), dataset.getYTest());
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}
