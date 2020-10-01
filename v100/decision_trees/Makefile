APP=cuda_cart

all:
	rm -rf build && mkdir build && cd build && cmake -GNinja ../ && ninja && cp ${APP} ../ && cd ../

clean:
	rm -rf build ${APP}
