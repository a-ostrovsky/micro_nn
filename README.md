# micro_nn

This is an educational library for solving neural networks. It is designed to be fully `constexpr` and hence to be runnable at compile time. The only purpose is to learn how neural networks are solved and what can be potentially `constexpr` in C++23.

This project is heavily work in progress and not finished yet.

## Currently implemented

✅ Some Layers (Linear, ReLU, Sigmoid)<br/>
✅ Some Loss Functions (Cross Entropy, MSE)<br/>
✅ SGD Optimizer<br/>

## TODO
⏳ Adam Optimizer<br/>
⏳ Solver<br/>
⏳ Data Loader<br/>

## Getting Started
```bash
git clone https://github.com/a-ostrovsky/micro_nn.git
mkdir build
cd build 
cmake ..
cmake --build .
ctest
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.