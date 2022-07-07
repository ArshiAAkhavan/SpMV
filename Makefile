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

bench_csr:
	mkdir -p bench/csr 
	gcc -fopenmp utils.c csr_parallel.c -DCOL_SIZE=10 -DROW_SIZE=10 -O3 -o bench/csr/parallel_10_10
	gcc -fopenmp utils.c csr_parallel.c -DCOL_SIZE=100 -DROW_SIZE=100 -O3 -o bench/csr/parallel_100_100
	gcc -fopenmp utils.c csr_parallel.c -DCOL_SIZE=1000 -DROW_SIZE=1000 -O3 -o bench/csr/parallel_1000_1000
	gcc -fopenmp utils.c csr_parallel.c -DCOL_SIZE=2000 -DROW_SIZE=1000 -O3 -o bench/csr/parallel_2000_1000
	gcc -fopenmp utils.c csr_parallel.c -DCOL_SIZE=4000 -DROW_SIZE=1000 -O3 -o bench/csr/parallel_4000_1000
	gcc -fopenmp utils.c csr_parallel.c -DCOL_SIZE=8000 -DROW_SIZE=1000 -O3 -o bench/csr/parallel_8000_1000
	gcc -fopenmp utils.c csr_parallel.c -DCOL_SIZE=8000 -DROW_SIZE=8000 -O3 -o bench/csr/parallel_8000_8000
	gcc utils.c csr_serial.c -DCOL_SIZE=10 -DROW_SIZE=10 -O3 -o bench/csr/serial_10_10
	gcc utils.c csr_serial.c -DCOL_SIZE=100 -DROW_SIZE=100 -O3 -o bench/csr/serial_100_100
	gcc utils.c csr_serial.c -DCOL_SIZE=1000 -DROW_SIZE=1000 -O3 -o bench/csr/serial_1000_1000
	gcc utils.c csr_serial.c -DCOL_SIZE=2000 -DROW_SIZE=1000 -O3 -o bench/csr/serial_2000_1000
	gcc utils.c csr_serial.c -DCOL_SIZE=4000 -DROW_SIZE=1000 -O3 -o bench/csr/serial_4000_1000
	gcc utils.c csr_serial.c -DCOL_SIZE=8000 -DROW_SIZE=1000 -O3 -o bench/csr/serial_8000_1000
	gcc utils.c csr_serial.c -DCOL_SIZE=8000 -DROW_SIZE=8000 -O3 -o bench/csr/serial_8000_8000

