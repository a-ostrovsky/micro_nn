# micro_nn

This is an educational library for solving neural networks. It is designed to be fully `constexpr` and hence to be runnable at compile time. The only purpose is to learn how neural networks are solved and what can be potentially `constexpr` in C++23. Probably it has no practical use.

This project is heavily work in progress and not finished yet.

Should work at least with
- *MSVC 19.40*
- *GCC 13.2*
- (Probably also with *Clang*, but not tested)

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

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
