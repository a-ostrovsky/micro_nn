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
✅ SGD optimizer<br/>
✅ Data loader<br/>
✅ Regularization<br/>
✅ Random number generation<br/>

## TODO
⏳ Adam optimizer<br/>
⏳ Improve compile speed <br/>
⏳ Data Loader -> Shuffle indices<br/>
⏳ Proper weight initialization (e.g., Kaiming)<br/>
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

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
