import subprocess
import time

def run_parallel_scripts():
    # 各スクリプトを並列実行
    command1 = ["mpiexec", "-n", "6", "python", "reghmc_parallel_.py"]
    command2 = ["python", "3d_new_notrem.py"]

    start_time = time.time()
    process1 = subprocess.Popen(command1)
    process2 = subprocess.Popen(command2)

    # 両方のプロセスが完了するのを待つ
    process1.wait()
    process2.wait()
    end_time = time.time()

    # 計算時間を計測
    duration = end_time - start_time
    print(f"Parallel scripts completed in {duration:.2f} seconds.")
    print("------------------------------------")

    return duration  # 並列処理全体の時間を返す

def execute_onlypotential():
    # `onlypotential.py` を実行
    command = ["python", "onlypotential_copy.py"]

    start_time = time.time()
    process = subprocess.Popen(command)
    process.wait()
    end_time = time.time()

    # 計算時間を計測
    duration = end_time - start_time
    print(f"`onlypotential_copy.py` completed in {duration:.2f} seconds.")
    print("-------------------------------------")

    return duration  # 実行時間を返す

if __name__ == "__main__":
    num_iterations = 20000  # 繰り返し回数を設定
    total_parallel_time = 0
    total_onlypotential_time = 0

    start_time = time.time()
    for i in range(num_iterations):
        print(f"Starting iteration {i + 1}")

        # 並列処理を実行
        parallel_time = run_parallel_scripts()
        total_parallel_time += parallel_time

        # onlypotential.py を実行
        onlypotential_time = execute_onlypotential()
        total_onlypotential_time += onlypotential_time

        print(f"Iteration {i + 1} completed.\n")

    end_time = time.time()
    total_time = end_time - start_time

    # 結果を出力
    print(f"Total parallel scripts time: {total_parallel_time:.2f} seconds.")
    print(f"Total `onlypotential.py` time: {total_onlypotential_time:.2f} seconds.")
    print(f"Overall total time: {total_time:.2f} seconds.")
    print("All iterations have completed.")

