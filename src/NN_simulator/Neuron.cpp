#include "Neuron.h"

using namespace std;

double Neuron::eta = 0.15; // net learning rate
double Neuron::alpha = 0.5; // momentum



Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}


double Neuron::activationFunction(double x)
{
    // tanh - output range [-1.0..1.0]

    return tanh(x);
}


double Neuron::activationFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}


void Neuron::calcOutputVal(const Layer &prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::activationFunction(sum);
}



void Neuron::calcOutputDeltas(double targetVal)
{
    double gradientComponent = m_outputVal- targetVal;
    m_nodeDelta = gradientComponent * Neuron::activationFunctionDerivative(m_outputVal);
}


double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed in next layer.

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_nodeDelta;
    }

    return sum;
}



void Neuron::calcHiddenDeltas(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_nodeDelta = dow * Neuron::activationFunctionDerivative(m_outputVal);
}


void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the node delta and train rate:
                (-1)*eta
                *neuron.getOutputVal()
                * m_nodeDelta
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

