all: dense_serial csr_serial csr_parallel csr_aligned_parallel ellpack_slice_simd

dense_serial: 
	mkdir -p build
	gcc dense/utils.c dense/dense_serial.c -O3 -o build/dense_serial

csr_serial: 
	mkdir -p build
	gcc csr/utils.c csr/csr_serial.c -O3 -o build/csr_serial

csr_parallel:
	mkdir -p build
	gcc -fopenmp csr/utils.c csr/csr_parallel.c -O3 -o build/csr_parallel

csr_aligned_parallel:
	mkdir -p build
	gcc -fopenmp csr/utils.c csr/csr_parallel_aligned.c -O3 -o build/csr_aligned_parallel

ellpack_slice_simd:
	mkdir -p build
	g++ -mfma -mavx2 simd/utils.cpp simd/ellpack_slice_simd.cpp -O3 -o build/ellpack_slice_simd
clean:
	rm -rf build
	rm -rf bench

bench: all
	python3 bench.py
