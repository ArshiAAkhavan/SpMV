import matplotlib.pyplot as plt
import subprocess
import re


def bench_specific(
    file_name: str, row_size: int, col_size: int, sparsity_factors: list[float]
) -> list[float]:
    results = []
    for s_factor in sparsity_factors:
        cmd = subprocess.Popen(
            [f"./build/{file_name}", str(row_size), str(col_size), str(s_factor)],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        last_line = cmd.stdout.readlines()[-1].decode("utf-8")
        cmd.communicate()
        elapsed_time = (int)(re.search("time: (\d*)", last_line).group(1))
        results.append(elapsed_time)
    return results


def bench_dense(
    row_size: int, col_size: int, sparsity_factors: list[float]
) -> list[float]:
    return bench_specific("dense_serial", row_size, col_size, sparsity_factors)


def bench_csr(
    row_size: int, col_size: int, sparsity_factors: list[float]
) -> dict[str, list[float]]:
    result = {}
    for csr_implementation in ["csr_serial", "csr_parallel", "csr_aligned_parallel"]:
        result[csr_implementation] = bench_specific(
            csr_implementation, row_size, col_size, sparsity_factors
        )
    return result


def bench_ellpack(
    row_size: int, col_size: int, sparsity_factors: list[float]
) -> list[float]:
    return bench_specific("ellpack_slice_simd", row_size, col_size, sparsity_factors)


def draw_plot(x, y, plot_name):
    plt.plot(x, y, label=plot_name)


row_size = 2000
col_size = 150000
sparsity_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]

print("drawing dense plot")
draw_plot(
    sparsity_factors,
    bench_dense(row_size, col_size, sparsity_factors),
    f"dense(M[{row_size}][{col_size}])",
)


for key, result in bench_csr(row_size, col_size, sparsity_factors).items():
    print(f"drawing csr_{key} plot")
    draw_plot(
        sparsity_factors,
        result,
        f"{key}(M[{row_size}][{col_size}])",
    )

print("drawing ellpack plot")
draw_plot(
    sparsity_factors,
    bench_ellpack(row_size, col_size, sparsity_factors),
    f"ellpack(M[{row_size}][{col_size}])",
)

plt.legend()
plt.xlabel("sparsity_factor(%)")
plt.ylabel("time(ms)")
plt.title("time/sparsity")
plt.show()
