# equation-learner

This is a reproduction of the EQL machine learning model and its variations [1] [2] using Keras and Tensorflow.

![Imgur](https://i.imgur.com/HZXwfVI.png)

## Getting Started

It's worth noting that the resources in this repository are not perfectly documented and also highly customized to the research of Elijah Sheridan. Use at your own risk.

That being said, this repository can be used via a cloning to either your working directory or the folder housing your downloaded Python packages. `models.py` houses the `EQL` and `EQLDIV` classes, the two ML models developed in this project. These classes depend upon Keras and Tensorflow, along with custom Keras Layer/Constraint/Regularizer classes which can be found in the `keras_classes.py` file. Example code for creating and training the `EQLDIV` model shown in the image above is as follows.

```
import equation_learner as eql

my_eql_model = eql.models.EQLDIV(inputSize=3, outputSize=3, numLayers=3)
my_eql_model.fit(predictors, labels, numEpoch=1000)
```

Both `EQL` and `EQLDIV` classes have utility methods which allow users to do things like print the equation which the model has learned (`getEquation()`), plot 2-D slices of that equation (`plotSlice()`), and obtain the number of active nodes in the model architecture (`sparsity()`).

The user can use the functions in the `data.py` file to generate `.npy` files for use as training/testing datasets.

If the user is training their model to learn systems of ODEs, they can use functions in the `ode.py` file to evaluate and visualize the accuracy of their trained models, along with various other attributes depending on the system (conservation of energy, etc).

`vpy.py` contains functions which utilize VPython to create animations of double pendulums, controlled by their true equations of motion and/or equations of motion learned by an EQL/EQL-div model.

## References

[1] Martius, Georg, and Christoph H. Lampert. "Extrapolation and learning equations." _arXiv preprint arXiv:1610.02995_ (2016).

[2] Sahoo, Subham S., Christoph H. Lampert, and Georg Martius. "Learning equations for extrapolation and control." _arXiv preprint arXiv:1806.07259_ (2018).
