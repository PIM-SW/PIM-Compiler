all: build

cmake:
	mkdir -p build && cd build && cmake -G Ninja ..;

build: cmake
	cd build && cmake --build . --target pim-compiler

clean_build: cmake clean
	cd build && cmake --build . --target pim-compiler

clean:
	rm -rf build
