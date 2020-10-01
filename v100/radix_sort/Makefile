APP=cuda_radix_sort

all:
	rm -rf build && mkdir build && cd build && cmake -GNinja ../ && ninja && cp ${APP} ../ && cd ../

clean:
	rm -rf build ${APP} cmake-build-release
