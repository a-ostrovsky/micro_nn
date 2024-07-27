[![Build](https://github.com/a-ostrovsky/micro_nn/actions/workflows/build.yml/badge.svg)](https://github.com/a-ostrovsky/micro_nn/actions/workflows/build.yml)

# micro_nn

This is an educational library for solving neural networks. It is designed to be fully `constexpr` and hence to be runnable at compile time. The only purpose is to learn how neural networks are solved and what can be potentially `constexpr` in C++23. Probably it has no practical use.

This project is heavily work in progress and not finished yet.

Should work at least with
- *MSVC 19.40*
- *GCC 13.2*
- *Clang 18.1*

## Example
This is an example how to create and train a neural network. It shows a simple linear regression y = 2x + 1
```cpp

// Initialize linear model of size 1x1
layers::Linear linear{1, 1};
linear.set_weights(linalg::Matrix<>{{{0.9f}}}, linalg::Matrix<>{{{0.9f}}});

// Create a 'sequential model' of the single linear model.
// Optimizer requires an input argument of type SequentialModel
model::SequentialModel model{std::move(linear)};

// We are using SGD optimizer here. 
// (Stochastic is just the name here. This example is deterministic.)
optimizer::SGDOptimizer optimizer{model};

// Use the Mean Square Error loss function for training evaluation.
loss::MSE<> mse{};

// Set-up training data
auto x1{linalg::Matrix<>{{{1.0}}}};
auto x2{linalg::Matrix<>{{{2.0}}}};
auto x3{linalg::Matrix<>{{{3.0}}}};

auto y1{linalg::Matrix<>{{{3.0}}}};
auto y2{linalg::Matrix<>{{{5.0}}}};
auto y3{linalg::Matrix<>{{{7.0}}}};

// Create a data loader to manage batches of data
// Here, we use a single batch containing all three data points
data::SimpleDataLoader dataloader(std::vector{x1, x2, x3},
                                  std::vector{y1, y2, y3}, 3);

// Train the model for 50 epochs.
Solver solver{model, optimizer, mse, dataloader};
solver.train(50);

// Predict the output for a new input feature using the trained model
auto y_pred{model.forward(linalg::Matrix<>{{{5.0}}})};
// Expected: The predicted value (y_pred.at(0, 0)) should be close to 11.0 after training

```

## Limitations
When evaluating models at compile time, ensure that the models are not too large and that training does not involve too many epochs. Exceeding the constexpr function evaluation step count limit will cause compilation to fail. To address this, you may need to adjust the step count limit for your compiler. The necessary options for some common compilers are listed in the table below.

| Compiler | Option                            |
|----------|-----------------------------------|
| GCC      | `-fconstexpr-ops-limit=100000000` |
| Clang    | `-fconstexpr-steps=100000000`     |
| MSVC     | `/constexpr:steps100000000`       |


## Currently implemented

✅ Some layers (Linear, ReLU, Sigmoid)<br/>
✅ Some loss functions (Cross Entropy, MSE)<br/>
✅ Optimizers: SGD, Adam<br/>
✅ Data loader<br/>
✅ Regularization<br/>
✅ Random number generation<br/>
✅ Weight initialization (Kaiming Normal)<br/>
✅ Learning Rate Scheduler (Step Decay)<br/>

## TODO
⏳ L2 regularization for the Adam optimizer.
⏳ Improve compile speed <br/>
⏳ More efficient matrix multiplication (e.g., Strassen)<br/>

## Getting Started
```bash
git clone https://github.com/a-ostrovsky/micro_nn.git
mkdir build
cd build 
cmake ..
cmake --build .
ctest
```

### Running with AddressSanitizer

To enable AddressSanitizer (ASAN) for detecting memory errors, use the `-DWITH_ASAN=ON` flag when configuring your project with CMake:

```bash
cmake -DWITH_ASAN=ON ..
```

## Interesting Language Aspects

### Stateful Metaprogramming

Inspired by [this article](https://mc-deltat.github.io/articles/stateful-metaprogramming-cpp20). This README outlines the basics. For more comprehensive details, see `unique_id.h`. Probably this is something which shouldn't be used in production. <br />

We can define a method, let's call it `counted_flag` which is declared but not yet defined. It is a friend method of `Reader<N>`. Every time when a specialization for `Writer<N>` is instantiated, the method is defined. Note, that it doesn't matter where the friend method is defined. It is okay to define it in a different struct like it is done here. Now when we have instantiated, e.g., `Writer<0>` and `Writer<1>`, then the methods `counted_flag` for `Reader<0>` and for `Reader<1>` will be defined while for `Reader<2>`, `Reader<3>` etc. won't. We can check whether the method definition already exists using `requires(Reader<N> r) { counted_flag(r); }`. With this building blocks, we can create a method which searches for the first `N` for which `counted_flag(Reader<N>)` is not yet defined, define it and return the value of `N`.
```cpp
template <unsigned N>
struct Reader {
    friend auto counted_flag(Reader<N>);
};
template <unsigned N>
struct Writer {
    friend auto counted_flag(Reader<N>) {}
    static constexpr unsigned n = N;
};
```

### Checking for constexpr'ness
Currently [std::sqrt](https://en.cppreference.com/w/cpp/numeric/math/sqrt) and [std::pow](https://en.cppreference.com/w/cpp/numeric/math/pow) are not `constexpr`. However, starting with C++26 they will be. Similarly, [std::abs](https://en.cppreference.com/w/cpp/numeric/math/abs) which will be `constexpr` in C++23. Therefore, it is useful to check whether a function is `constexpr` or not. This is implemented in `meta.h` as follows:
```cpp
// Implementation
template <class FuncT, void* = FuncT{}()>
constexpr bool is_constexpr(FuncT) { return true; }
constexpr bool is_constexpr(...) { return false; }

// Usage
if constexpr (meta::is_constexpr([&] { return std::abs(T{}); })) {
    return std::abs(t);
} else {
    return t < T{} ? -t : t;
}
```
This works because, in order for a lambda to be used as a template parameter, it must be `constexpr`. If the function which was called in the lambda is not constexpr, then the `is_constexpr` function returning true cannot be called due to SFINAE (Substitution Failure Is Not An Error). Consequently, the implementation with variadic parameters, will be called returning false. 


## Implemented Functionality

### Simple Linear Algebra Library
There is a simple linear algebra library implemented in the `Matrix` class template defined in `matrix.h` file. A `Matrix` can represent a 2D matrix or 1D vector. A `Matrix` supports arithmetic operations, including broadcasting for addition and subtraction. Additionally there are other operations, like transposition, applying a functor on each element or element-wise multiplication. The decision to avoid using external libraries like Eigen was made to achieve a fully `constexpr` implementation.

### Layers
This project implements Linear, ReLU and Sigmoid Layer which are defined in `layers.h`. The layers satisfy the concept `Layer` defined in the same file, so more layer types can be added easily.
* **Linear Layer** represents a fully connected neural network layer. It performs linear transformation of the input data. The forward pass computes $output = x \cdot weights + bias$.
* **Sigmoid Layer** applies the sigmoid activation function to each element of the input matrix. The forward pass computes $\frac{1}{1 + e^{-x}}$.
* **ReLU (Rectified Linear Unit) Layer** applies the ReLU activation function to each element of the input matrix. The forward pass computes $\max(0, x)$.

### Sequential Model
Only the sequential model is implemented. It represents a linear stack of layers where the output of one layer if the input to the next one.

### Dataloader
Dataloader loads the data which consists of the two vectors. The vector `x` holds the input features while the vector `y` holds the target output. The data can be partitioned in batches. Also there is support for shuffling data to avoid learning patters from order of the data.

### Loss function
Two loss functions are implemented in `loss.h`: MSE (Mean Square Error) and Cross Entropy Loss. The loss functions satisfy the `Loss` concept defined in the same file.
* **MSE** can be used for regression tasks. It calculates the mean squared errors between the true values and predicted values. The differences between those values are squared and added. Then the average is calculated: $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2$
* **Cross Entropy Loss** can be used for classification tasks. It is defined as $H(y_{\text{true}}, y_{\text{pred}}) = -\sum_{i} y_{\text{true}, i} \log(y_{\text{pred}, i})$

### Initialization
Kaiming Normal initialization is supported. It initializes the weights of a network with ReLU (Rectified Linear Unit) activation. This is done by setting the weights from normal distribution with mean of 0 and standard deviation of $\sqrt{\frac{2}{n}}$ where $n$ is the number of inputs to the layer. This approach helps to avoid vanishing or exploding gradients, especially if networks grow deeper.

### Optimization
This project implements the SGD (Stochastic Gradient Descent) optimizer. Following features are supported by the optimizer:
* **Weight decay** is a regularization parameter which penalizes large weights in order to avoid overfilling. For example, in a dataset predicting house prices with features like location, size, and age, a model might overfit by only considering the size, ignoring other features. Regularization can mitigate this by penalizing the model's complexity, encouraging a more balanced use of all features.
* **Momentum** helps to smooth out the updates and improve the convergence speed. This is done by adding a fraction of the previous update step to the current update step. When the previous update steps were updating the weights in same direction the momentum is accumulated. When the subsequent update is in a different direction, it is smoothed out. <br />
There is a step decay learning rate scheduler implemented in `lr_scheduler.h`. It decreases the learning rate by a factor every `N` epochs. This can help to converge faster. In the beginning the learning rate will be high to allow fast convergence. After some time it goes down to prevent overshooting the minimum.

### Training Solver
The `Solver` class, defined in `solver.h` is orchestrating the training process. It iterates over the dataset the specified number of epochs. It performs the forward pass, calculates the loss, performs the backward pass to calculate the gradients and the updates models weights.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
