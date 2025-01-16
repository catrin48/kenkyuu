# ファイル名
data_file = "swap_300.0_350.0.dat"  # 縦に0と1が記載されたファイル

# カウント用の変数
count_0 = 0
count_1 = 0

# ファイルを開いてデータを読み取る
with open(data_file, "r") as file:
    for line in file:
        value = line.strip()  # 行末の改行を削除
        if value == "0":
            count_0 += 1
        elif value == "1":
            count_1 += 1
total=count_0+count_1
acceptance=count_1/total
# 結果を出力
print(f"0の数: {count_0}")
print(f"1の数: {count_1}")
print(acceptance)
