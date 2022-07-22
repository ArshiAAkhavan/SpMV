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
	g++ -mfma -mavx2 simd/utils.cpp simd/ellpack_slice_simd_256.cpp -O3 -o build/ellpack_slice_simd_256
	g++ -mfma -mavx512vl -mavx512f simd/utils.cpp simd/ellpack_slice_simd_512.cpp -O3 -o build/ellpack_slice_simd_512
clean:
	rm -rf build
	rm -rf bench

bench: bench_rows bench_sparsity

bench_rows: all
	python3 bench_rows.py

bench_sparsity: all
	python3 bench_sparse.py
