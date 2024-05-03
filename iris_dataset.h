#ifndef IRIS_DATASET_H
#define IRIS_DATASET_H

#include <string>
#include <vector>
#include <tuple>

class IrisDataset
{
private:
    std::string filename;
    std::vector<std::vector<double>> X;
    std::vector<int> y;

public:
    IrisDataset(const std::string &filename);
    void load();
    void preprocess();
    std::tuple<std::vector<std::vector<double>>, std::vector<int>,
               std::vector<std::vector<double>>, std::vector<int>>
    trainTestSplit(double testSize);
    std::vector<std::vector<double>> getXTrain();
    std::vector<int> getYTrain();
    std::vector<std::vector<double>> getXTest();
    std::vector<int> getYTest();
};

#endif
