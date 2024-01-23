# PIM-Compiler

## Clang-based frontend to convert PIM-API to MLIR 

### Set up Polygeist [[1]](#1) with patch
```
  git submodule update --init --recursive
  cd Polygeist
  git am ../pim-avail.patch
```
### Build LLVM, MLIR, Clang
```
mkdir Polygeist/llvm-project/build && cd Polygeist/llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir  # can be skipped
```
### Build necessary pim-compiler libraries
```
mkdir PIM-Compiler/build && cd PIM-Compiler/build
cmake -G Ninja .. -DMLIR_DIR=$PWD/../../Polygeist/llvm-project/build/lib/cmake/mlir
cmake --build . --target CallToPIM
```
### Build mlir-clang with pim-compiler
```
mkdir Polygeist/build && cd Polygeist/build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
cmake --build . --target mlir-clang
```

To test,
```
./Polygeist/build/mlir-clang/mlir-clang --raise-scf-to-affine --pim-avail -S --function=cblas_saxpy PIM_BLAS/src/cblas_saxpy.c
```

## An out-of-tree MLIR dialect for supporting PIM

This setup assumes that you have downloaded and built LLVM and MLIR at your home directory. 
Paths are initally configured as follows.

LLVM_DIR=Polygeist/llvm-project/build/lib/cmake/llvm
MLIR_DIR=Polygeist/llvm-projectbuild/lib/cmake/mlir

To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja ..
cmake --build . --target pim-compiler
```
or just type
```sh
make
```

after build, test with following command
```sh
./build/pim-compiler/pim-compiler test/pnm.mlir --convert-pim
```

if you want instruction format, 
```sh
./build/pim-compiler/pim-compiler test/pnm.mlir --convert-pim > out.mlir
python3 translate.py 
```

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```

## References
<a id="1">[1]</a> 
Moses, William S., et al. "Polygeist: Affine C in MLIR." (2021).


