import matplotlib.pyplot as plt
import subprocess
import re


def bench_specific(file_name: str, row_size: int, col_sizes: list[int]) -> list[float]:
    results = []
    for col_size in col_sizes:
        cmd = subprocess.Popen(
            [
                f"./build/{file_name}",
                str(row_size),
                str(col_size),
                str(sparsity_factor),
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        last_line = cmd.stdout.readlines()[-1].decode("utf-8")
        cmd.communicate()
        elapsed_time = (int)(re.search("time: (\d*)", last_line).group(1))
        results.append(elapsed_time)
    return results


def bench_dense(row_size: int, col_sizes: list[int]) -> list[float]:
    return bench_specific("dense_serial", row_size, col_sizes)


def bench_csr(row_size: int, col_sizes: list[int]) -> dict[str, list[float]]:
    result = {}
    for csr_implementation in ["csr_serial", "csr_parallel", "csr_aligned_parallel"]:
        result[csr_implementation] = bench_specific(
            csr_implementation, row_size, col_sizes
        )
    return result


def bench_ellpack(row_size: int, col_sizes: list[int]) -> list[float]:
    return bench_specific("ellpack_slice_simd", row_size, col_sizes)


def draw_plot(x, y, plot_name):
    plt.plot(x, y, label=plot_name)


row_size = 1000
sparsity_factor = 0.5
col_sizes = [10000, 30000, 60000, 128000, 150000]

print("drawing dense plot")
draw_plot(
    col_sizes,
    bench_dense(row_size, col_sizes),
    f"dense(number of cols={row_size})",
)


for key, result in bench_csr(row_size, col_sizes).items():
    print(f"drawing csr_{key} plot")
    draw_plot(
        col_sizes,
        result,
        f"{key}(number of cols={row_size})",
    )

print("drawing ellpack plot")
draw_plot(
    col_sizes,
    bench_ellpack(row_size, col_sizes),
    f"ellpack(number of cols={row_size})",
)

plt.legend()
plt.xlabel("number of rows")
plt.ylabel("time(ms)")
plt.title("time/number of rows")
plt.show()
