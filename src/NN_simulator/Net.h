#pragma once
#include <vector>

using namespace std;

class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getNetError(void) const { return m_netError; }

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_netError; 

};