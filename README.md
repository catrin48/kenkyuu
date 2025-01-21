研究に使用しているプログラム
必要なパッケージ
amber,
温度レプリカ交換法、力場レプリカ交換法、HMC法
HMC法によるフォールディングについて
HMC法の時間発展部分と採択部分で異なるモデルを使用するHMC法をMLHMC法。
系のサンプリングが決定される採択部分のモデルを高精度。
時間発展部分は近似的だが、計算コストが小さいので、結果的に高効率かつ高精度なシミュレーションが可能と考える。
より良くするために、HMC法とMLHMC法で構造交換かつ温度レプリカ交換法を行う。
src 
3d_new_notrem.py 3D-RISMによる温度レプリカ交換法
forcefields.py 力場関連
functions.py 関数
methods.py 
reghmc_paralled.py GB/SA法を用いた温度レプリカ交換法
ffremtrem.py 実行