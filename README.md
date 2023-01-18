Extreme Learning Machine(ELM): Python code
===

ELM was originally proposed to train "generalized" single-hidden layer feedforward neural networks(SLFNs) with fast learning speed, good generalization capability and provides a unified learning paradigm for regression and classification. This project implemented the ELM  algorithm with python 3.5, you can download source code and install it.

Installation
---

Download source code, enter cmd in the project directory, run the following command:</br>

```c
python setup.py install
```

Then you can import elm module in python like following code:

```python
import elm
```

About ELM
---

The structure of ELM is shown in following figure.

![](https://github.com/5663015/elm/blob/master/picture/structure_of_ELM.jpg)

ELM does not need BP algorithm to train the network. First, randomly initialize the input layer to the hidden layer weight, then directly calculate the hidden layer to the output layer weight matrix beta. 
The output of ELM with L hidden nodes can be written as:

![](https://github.com/5663015/elm/blob/master/picture/output.jpg)

Only the weights between the hidden layer and the output, beta matrix, need to be deterimined. The aim is to minimize following formulation:

![](https://github.com/5663015/elm/blob/master/picture/aim_function.jpg)

If there is no regularization C, the solution of beta is:

![](https://github.com/5663015/elm/blob/master/picture/no_re.jpg)

If set regularization factor C, the solutions of beta are:

* the number of training samples is not huge(solution 1):

![](https://github.com/5663015/elm/blob/master/picture/solution1.jpg)

* the number of training samples is huge(solution 2):

![](https://github.com/5663015/elm/blob/master/picture/solution2.jpg)


API
---

`class elm.elm(hidden_units, activation_function,  x, y, C, elm_type, one_hot=True, random_type='normal')`

* hidden_units: list, shape [hidden units, output units], numbers of hidden units and output units
* activation_function: str, 'sigmoid', 'relu', 'sin', 'tanh' or 'leaky_relu'. Activation function of neurals
* x: array, shape[samples, features]. The input of neural network.
* y: array, shape[samples, ], labels
* C: float, regularization parameter
* elm_type: str, 'clf' or 'reg'. 'clf' means ELM solve classification problems, 'reg' means ELM solve regression problems.
* one_hot: bool, Ture or False, default True. The parameter is useful only when elm_type == 'clf'. If the labels need to transformed to one_hot, this parameter is set to be True.
* random_type: str, 'uniform' or 'normal', default:'normal'. Weight initialization method

`elm.elm.fit(algorithm)`

Train the model, compute beta matrix, the weight matrix from hidden layer to output layer

* Parameter:</br>
  * algorithm: str, 'no_re', 'solution1' or 'solution2'. The algorithm to compute beta matrix
* Return:</br>
  * beta: array. The weight matrix from hidden layer to output layer
  * train_score: float. The accuracy or RMSE
  * train_time: str. Time of computing beta

`elm.elm.predict(x)`

Compute the result given data

* Parameter:
  * x: array, shape[samples, features]
* Return:
  * y_: array. Predicted results

`elm.elm.score(x, y)`

Compute accuracy or RMSE given data and labels

* Parameters:
  * x: array, shape[samples, features]
  * y: array, shape[samples, ]
* Return:
  * test_score: float, accuracy or RMSE

CABLE FAULTS USING LEARNING ALGORITHM IN LARGE SCALE SOLAR (LSS)
===
Fault detection and timely troubleshooting are essential for the optimum performance in any power generation system, including photovoltaic (PV) systems. In particular, the goal for any commercial power-producing house is maximizing power production, minimizing energy loss and maintenance cost, and the safe operation of the facility. Since PV systems are subject to various faults and failures, early detection of such faults and failures is very crucial for achieving the goal 

With the large numbers of PV modules and strings, it would be a problem to detect the faulty string or PV module in a short time duration. Troubleshooting requires the fault finder team to shut down certain strings to locate the fault which is obviously based on trial and error method. Once the PV is shutdown, it will take 30 minutes to resume the power supply. Thus, more hours are needed if the trial and error method begins at a CB which is far from the actual fault location.  Hence, a proper method is crucial in this case to locate the affected PV string so that the fault can be detected and fixed quickly to resume the operation of PV farm. 

By developing a fault diagnosis model for fault detection and localisation of DC cable in PV systems using machine learning technique this would help the time taken for the remediation process shorten. Thus, the power plan can resume its operation in an immediate manner.



References
---

[1] Huang G B, Zhu Q Y, Siew C K. Extreme learning machine: theory and applications[J]. Neurocomputing, 2006, 70(1-3): 489-501.</br>
[2] Huang G B, Zhou H, Ding X, et al. Extreme learning machine for regression and multiclass classification[J]. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 2012, 42(2): 513-529.</br>
[3] http://www.ntu.edu.sg/home/egbhuang/index.html</br>
[4] X. Li, W. Li, Q. Yang, W. Yan and A. Y. Zomaya, "An Unmanned Inspection System for Multiple Defects Detection in Photovoltaic Plants," in IEEE Journal of Photovoltaics, vol. 10, no. 2, pp. 568-576, March 2020, doi: 10.1109/JPHOTOV.2019.2955183.</br>
[5] A. Dhoke, R. Sharma and T. K. Saha, "Condition monitoring of a large-scale PV power plant in Australia," 2016 IEEE Power and Energy Society General Meeting (PESGM), Boston, MA, 2016, pp. 1-5, doi: 10.1109/PESGM.2016.7742048.</br>





