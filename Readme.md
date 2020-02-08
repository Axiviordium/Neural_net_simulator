# Симулятор нейросети на C++

Данный репозиторий содержит разбор реализации симулятора многослойной нейронной сети. При создании сети возможно задать количество слоев и число нейронов в слоях. В качестве примера реализована сеть, моделирующая поведение функции XOR (исключающее или).
***
## Оглавление
1. [Введение](#Введение)
2. [Модель ИНС](#Модель-ИНС)
3. [Алгоритм обратного распространения ошибки][Backprop_pdf]
   + Выходной слой
   + Скрытые слои
   + Краткий итог
4. [Реализация](#Реализация)
   + [Детали реализации](#Детали-реализации)
   + [class Net](#class-Net)
   + [class Neuron](#class-Neuron)
   + [class TrainingData](#class-TrainingData)
   + [Генератор данных](#Генератор-данных)


[Backprop_pdf]: Backpropagation.pdf

***
## **Введение**
***

**`Искусственная нейронная сеть (ИНС)`** - математическая модель взаимодействия искусственных нейронов и её программная или аппаратная реализация.

Нейронные сети не программируются в привычном смысле этого слова, они обучаются. Возможность обучения — одно из главных преимуществ нейронных сетей перед традиционными алгоритмами. 
Технически обучение заключается в нахождении коэффициентов связей между нейронами. В процессе обучения нейронная сеть способна выявлять сложные зависимости между входными данными и выходными, а также выполнять обобщение. 
Это значит, что в случае успешного обучения сеть сможет вернуть верный результат на основании данных, которые отсутствовали в обучающей выборке, а также неполных и/или «зашумленных», частично искажённых данных.

Нейросеть может формироваться различным количеством нейронов, при этом нейроны могут группироваться в слои. По количеству слоёв сети можно разделить на:
+ `однослойные` - сети, в которых сигналы от входного слоя сразу подаются на выходной 
+ `многослойные` - сети, состоящие из входного, выходного и некоторого количества скрытых слоёв.

*Многослойные* сети могут быть:
+ `сетями прямого распространения` (*`feed-forward network`*) - сети,  в которых сигнал распространяется строго от входного слоя к выходному. Такая конструкция имеет соединения нейронов только в одном направлении, т.е. формирует направленный ациклический граф. Сеть прямого распространения является функцией своих входов, не имеет внутреннего состояния, кроме весов связей. Таким образом ответ сети не зависит от предыдущих данных, поступивших на вход.
+ `сетями с обратной связью` (*`recurrent network`*) - сети, в которых выход нейрона может вновь подаваться на его вход. В таких сетях уровень активности нейронов не статичен и может приводит как к стабильному состоянию, так и к хаотичному поведению. Ответ сети на определённые значения на входе зависит от внутреннего состояния, на которое влияют предыдущие входные значения. Таким образом сети с обратной связью могут проявлять кратковременную память (в отличие от сетей прямого распространения). 


***
## **Модель ИНС**
***
Рассмотрим модель простой нейронной сети. (см.рис.1). 
Сеть состоит из трёх слоёв: 
+ входного слоя
+ одного скрытого слоя
+ выходного слоя (в данном случае состоящего из одного нейрона)

Количество нейронов в каждом слое может быть любым.

![Простая трёхслойная сеть][Pic_1]

[Pic_1]: pics/Pic_1.bmp
Рис.1 Простая трёхслойная сеть


***Входной слой***

`Нейроны входного слоя` получают некоторые значения исходных данных и без изменений передают их нейронам скрытого слоя. Каждый нейрон входного слоя на рис.1 соединён с каждым нейроном скрытого слоя. Это не является обязательным условием. Однако, в данной работе, для простоты, примем следующие допущения:
+ `все` нейроны `предшествующего` слоя имеют связи `со всеми` нейронами `следующего` слоя
+ `слой`, расположенный левее по картинке, `может соединяться лишь со следующим слоем` расположенным правее (т.е. невозможно, чтобы связи из нейронов уходили в какой-либо из предыдущих слоёв или в какой-либо последующий слой, кроме ближайшего правого) 

Таким образом в дальнейшем мы рассматриваем полносвязную сеть с прямым распространением.


***Скрытые слои***

Количество скрытых слоёв может быть любым.
`Нейрон скрытого слоя` суммирует `взвешенные` значения нейронов предыдущего слоя и применяет к полученной сумме `функцию активации` (см. рис.2) ("+" - означает суммирование, "f" - применение функции активации)

![Нейрон скрытого слоя][Pic_2]

[Pic_2]: pics/Pic_2.bmp
Рис.2 Функции нейрона скрытого слоя 


***Выходной слой***

В данной модели `нейроны выходного слоя` работают подобно нейронам скрытого слоя. Отличие в том, что у выходного слоя нет связей со следующим слоем.  После применения функции активации к взвешенной сумме сигналов, `нейрон выходного слоя` получает конечное значение, ответ.   


***Нейроны смещения***

Нам потребуется ещё один вид нейронов (см рис.3).

![Трёхслойная сеть с нейронами смещения][Pic_3]

[Pic_3]: pics/Pic_3.bmp
Рис.3. Добавление нейронов смещения

В каждом слое добавим по одному `нейрону смещения` (*`bias neuron`*). 
 Нейрон смещения:
 + не имеет входов
 + выходное значение всегда  равно 1
 + веса связей нейрона смещения с другими нейронами могут изменяться

[К оглавлению](#Оглавление)

***

## **Реализация**
***

Приближённый алгоритм работы с ИНС:

+ задаём `топологию` (конфигурацию) сети 
+ создаём сеть
+ выполняем обучение сети на исходных данных:
  +  `вычисление выходных данных слоёв` - передавая входные значения первому слою, после прохождения сигнала через слои сети получаем некоторый ответ на выходном слое  
  + `вычисление векторов ошибок слоёв` - сравнивая полученный ответ с ожидаемым корректируем веса связей между нейронами
  + `изменение весовых коэффициентов`
  + `проверка условия окончания обучения` - переходим к следующему набору исходных данных или завершаем обучение



Для реализации описанной модели создадим классы `Net`, `Neuron`.
Диаграмма классов имеет следующий вид (см. рис.8)

![Диаграмма классов][Pic_8]

[Pic_8]: pics/Pic_8.bmp
Рис.8. Диаграмма классов
***
### ***Детали реализации***:
+ `Топологию` сети будем описывать с помощью `вектора беззнаковых целых` чисел.  
   + Каждое `число`  в векторе `соответсвует количеству нейронов` в соответствующем слое `без учёта нейрона смещения`.
   + Например, если в объекте topology сохранены числа {3 7 9 4 1}, то во входном слое будет 3 нейрона + 1 нейрон смещения (фактически 4), в первом скрытом слое - 7 нейронов +1, во втором скрытом - 9 обычных + 1 смещения, в третьем скрытом - 4+1, а в выходном - 1+1 нейрон.
+ Набор `входных` значений будем хранить с помощью `вектора вещественных` чисел.
+ Набор `выходных` значений будем хранить с помощью `вектора вещественных` чисел.
+ определим тип `Layer`, как вектор, содержащий объекты  класса neuron. Таким образом, каждый слой будет представлен отдельным объектом.
```c++
typedef vector<Neuron> Layer;
```
+ определим тип `Connection`, как пару вещественных чисел, хранящих значение веса и изменение этого значения для текущей связи нейрона с нейроном из следующего слоя.
```c++
struct Connection
{
    double weight;
    double deltaWeight;
};
```
+  Все скрытые члены данных будем начинать с префикса `m_`

***
### ***class Net***
Класс net описывает всю сеть, как единое целое и предлагает методы для работы с ней:


```c++
class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
   
private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_netError; 

};
```
+ *Конструктор класса Net*

Чтобы создать объект класса Net, а именно сеть с определённым количеством слоёв и определённым количеством нейронов, мы передаём в конструктор ссылку на объект (&topology), хранящий топологию сети. Ранее мы условились, что нейроны будут формировать слои и каждый слой будет отдельным объектом. Тогда вся сеть станет представлять собой массив слоёв. Этот массив будет содержаться в переменной `m_layers` (см. рис.9).

![ИНС в виде массива слоёв][Pic_9]

[Pic_9]: pics/Pic_9.bmp
Рис.9. Вектор слоёв

```c++
Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron!" << endl;
        }

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        m_layers.back().back().setOutputVal(1.0);
    }
}
```
Переменная `numOutputs` равна количеству выходящих `связей` из нейрона. Выходной слой последний, поэтому numOutputs=0, для других слоёв numOutputs равна количеству нейронов в ближайшем справа слое.  Это значение используется при создании объекта класса `Neuron`, `class Neuron` будет описан [позже](#class-Neuron).

Внутренний цикл заполняет слой нейронами, с индексами от 0 до индекса равного числу, указанному в топологии. Таким образом в каждый слой будет включён дополнительный нейрон смещения. С помощью метода `setOutputVal()` класса `Neuron` мы устанавливаем выходное значение нейрона смещения (он  последний в векторе) равное 1.

+ *Метод feedForward() класса Net*
```c++
void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].calcOutputVal(prevLayer);
        }
    }
}
```
Перед выполнением функции проверяем, равно ли число входных значений (количество элементов исходных данных) числу входных нейронов (за вычетом нейрона смещения).
Выставляем значения входных нейронов. Затем, поочерёдно для каждого слоя (после входного) вызываем функцию-член `calcOutputVal() класса Neuron` у каждого нейрона в слое (за вычетом нейрона смещения).

+ *Метод backProp() класса Net*

Каждому набору входных данных соответствует набор целевых значений (правильных ответов). Ссылка на этот набор целевых значений передаётся в метод.

```c++
void Net::backProp(const vector<double> &targetVals)
{
    // Calculate overall net error 

    Layer &outputLayer = m_layers.back();
    m_netError = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double neuronError = targetVals[n] - outputLayer[n].getOutputVal();
        m_netError += neuronError * neuronError;
    }
    m_netError /= 2; // get average error squared
  

    // Calculate output layer node deltas

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputDeltas(targetVals[n]);
    }

    // Calculate hidden layer node deltas

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenDeltas(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}
```


После получения ответа на выходных нейронах для определённого набора входных значений, вычисляем значение ошибки сети (целевой функции) и сохраняем в переменной `m_netError`.
Затем вычисляем ошибки в узлах (`дельта`) для выходного и скрытых слоёв, используя методы `calcOutputDeltas()` и `calcHiddenDeltas()` класса Neuron. 

*Примечание* Скрытые слои имеют индексы от [1] до [m_layers.size()-2]. Входной слой имеет индекс m_layers[0], выходной - m_layers[m_layers.size()-1].

 После вычисления градиентов обновляем веса связей у всех нейронов с помощью метода `updateInputWeights()` класса Neuron. Обновление выполняется начиная от выходного слоя и заканчивается первым(самым левым) скрытым, т.к. у входного слоя нет весов на входах.


+ *Метод getResults() класса Net*

Метод для считывания выходных значений нейронов выходного слоя.
```c++
void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}
```

[К оглавлению](#Оглавление)
***
### ***class Neuron***
```c++
class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void calcOutputVal(const Layer &prevLayer);
    void calcOutputDeltas(double targetVal);
    void calcHiddenDeltas(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;   // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_nodeDelta;
};
```
Связи нейрона с нейронами следующего слоя хранятся в векторе `vector<Connection> m_outputWeights`. Первый элемент в векторе - связь нейрона с первым элементом следующего слоя, второй - со вторым и т.д.


+ *Конструктор класса Neuron*

```c++
Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}
```
Конструктор  по умолчанию объекта Connection инициализирует его поля 0.


+ *Метод calcOutputVal() класса Neuron*

```c++
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
```
`prevLayer[n].m_outputWeights[m_myIndex].weight` - вес связи нейрона (с индексом `n`) из предыдущего слоя с текущим нейроном (индекс текущего `m_myIndex`) 

+ *Метод activationFunction() класса Neuron*



```c++
double Neuron::activationFunction(double x)
{
    // tanh - output range [-1.0..1.0]

    return tanh(x);
}
```

+ * Метод activationFunctionDerivative() класса Neuron*

Ввиду особенностей принятой функции активации (гиперболический тангенс) для нахождения производной достаточно знать значение на выходе нейрона после активации, поэтому в функцию будем передавать значение `m_outputVal` (которое и есть значение на выходе нейрона после активации)



```c++
double Neuron::activationFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}
```
+ *Метод calcOutputDeltas() класса Neuron*

См. подраздел "Краткий итог" раздела "Алгоритм обратного распространения ошибки"
```c++
void Neuron::calcOutputDeltas(double targetVal)
{
    double gradientComponent = m_outputVal- targetVal;
    m_nodeDelta = gradientComponent * Neuron::activationFunctionDerivative(m_outputVal);
}
```
+ *Метод calcHiddenDeltas() класса Neuron*

См. подраздел "Краткий итог" раздела "Алгоритм обратного распространения ошибки"

Как и планировали нахождение ошибки (дельта) в скрытых узлах разбиваем на 2 отдельные функции: 
+ `sumDOW()` - одна вычисляет сумму произведений ошибок(дельт) узлов следующего слоя (связанных с текущим узлом) и весов связей 
+ `calcHiddenDeltas()` - другая вычисляет произведение полученной суммы и производной от функции активации


```c++
void Neuron::calcHiddenDeltas(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_nodeDelta = dow * Neuron::activationFunctionDerivative(m_outputVal);
}
```
+ *Метод sumDOW() класса Neuron*

```c++
double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed in next layer.

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_nodeDelta;
    }

    return sum;
}
```
+ *Метод updateInputWeights() класса Neuron*

См. подраздел "Краткий итог" раздела "Алгоритм обратного распространения ошибки"

```c++
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
```

### ***class TrainingData***
Для работы с набором данных создадим вспомогательный класс TrainingData


```c++
class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};
```

+ *конструктор класса TrainingData*

```c++
TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}
```

+ *Метод getTopology() класса TrainingData*

```c++
void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}
```
+ *Метод getNextInputs() класса TrainingData*

```c++
unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}
```

+ *Метод getTargetOutputs() класса TrainingData*

```c++
unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}
```
### ***Генератор данных***

Создание набора данных выполним с помощью генератора данных. Примером будет набор данных соответствующих работе функции XOR. Данные должны быть сформированы по определённому шаблону. Первая строка должна задать топологию создаваемой сети. В остальных - содержаться входные и целевые значения.  

```c++
int main()
{
	std::cout << "topology: 2 3 1" << std::endl;
	for (int i = 2000; i >= 0; i--)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
		int t = !(n1 & n2);

		std::cout << "in: " << n1 << ".0 " << n2 << ".0" << std::endl;
		std::cout << "out: " << t << ".0 " << std::endl;
	}
}
```
