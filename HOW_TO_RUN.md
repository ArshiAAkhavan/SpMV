# Build 

## Build all
```bash 
make
```

## Build specific target
```bash 
make dense_serial
make csr_serial
make csr_parallel
make csr_aligned_parallel
make ellpack_slice_simd
```

# Clean up 
```bash 
make clean
```

# Run bench marks
Make sure that you have `mathplotlib` python module installed.

## Run all benchmarks
```bash 
make bench
```

## Run specific benchmark
```bash 
make bench_rows
make bench_sparsity
```
