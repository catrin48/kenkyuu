u"""
レプリカ交換の解析ためのクラス.

連番ファイルの形式 「名前00.拡張子」

指数和の計算について
    普通に計算すると発散する危険性があるので、対数をとって以下の関係式を用いている.
    log[Σ{a_i}] = max[{log(a_i)}] + log[Σexp{log(a_i)-max({log(a_i)})}]
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt


class REM:
    u"""レプリカ交換に関するツールを管理."""

    def __init__(self, fname, k_boltz):
        u"""
        各レプリカの温度を取得.

        Parameters
        ----------
        fname : str
            レプリカの温度データファイル名
            温度の数値が書かれた1列のみのファイルにしておくこと!
        k_boltz : float
            エネルギーの単位に合わせたボルツマン定数の値
        """
        self.k_boltz = k_boltz  # ボルツマン定数

        # 各レプリカの温度と逆温度をNumpy配列で取得
        temp = []
        with open(fname, "r") as f:
            for line in f:
                temp += [float(line)]
        self.nrep = len(temp)
        self.temp = np.array(temp)
        self.beta = 1.0 / (k_boltz * self.temp)

    def remake_file(self, data_name, trajec_name, output_name=None, split=" "):
        u"""
        構造固定で出力されたファイルを，温度固定に書き換え直す.

        Parameters
        ----------
        data_name : str
            書き換えるファイル
        trajec_name : str
            レプリカ交換の軌跡
        output_name : str
            書き換えたファイルの名前(デフォルト有り)
        """
        trajec_file = open(trajec_name, "r")
        name, ext = os.path.splitext(data_name)
        data_file = [open(name + f"{irep + 1:02d}" + ext, "r")
                     for irep in range(self.nrep)]
        if output_name is None:
            output_name = name + "_temp" + ext
        name, ext = os.path.splitext(output_name)
        output_file = [open(name + f"{irep + 1:02d}" + ext, "w+")
                       for irep in range(self.nrep)]

        while(True):
            line = trajec_file.readline()
            if not line:
                break
            permut = list(map(int, line.rstrip().split(split)))[1:]
            data = [data_file[irep].readline() for irep in range(self.nrep)]
            for irep, permut_irep in enumerate(permut):
                output_file[permut_irep].write(data[irep])

        for irep in range(self.nrep):
            data_file[irep].close()
            output_file[irep].close()
        trajec_file.close()

    def get_roundtrip_rate(self, f_in, split=" "):
        u"""
        レプリカ間の往復率を計算する(多少の誤差は気にしない).

        Parameters
        ----------
        f_in : str
            レプリカ交換の軌跡が書かれたファイル名
        split : str
            データの区切り文字

        Returns
        -------
        roundtrip : float
            往復率
        """
        nrep = self.nrep
        trajectory = []

        # 軌跡の取得(ステップの列は切る)
        with open(f_in, "r") as file:
            for line in file:
                trajectory +=\
                    [[int(s) for s in line.rstrip().split(split)[1:]]]

        # 往復を判定するフラグの作成
        flag = [(0, nrep - 1)] * nrep

        roundtrip = 0.0
        for trajec_i in trajectory:  # 軌跡のループ
            for idx, replica in enumerate(trajec_i):   # レプリカのループ

                if replica in flag[idx]:  # レプリカが端についたときの処理
                    roundtrip += 1.0
                    if replica == 0:
                        flag[idx] = (nrep - 1,)
                    else:
                        flag[idx] = (0,)

        roundtrip /= len(trajectory) * nrep * 2
        return roundtrip

    def get_multi_histogram(self, f_in, column, nhist=100, split=" ",
                            hmin=None, hmax=None, f_out="hist_rem.dat",
                            probability=False, plot=True):
        u"""
        各レプリカのエネルギーヒストグラムを取得.

        Parameters
        ----------
        f_in : str
            エネルギー瞬間値のファイル名(連番数字は抜いて与える)
        column : int
            エネルギーが何列目か(0から数える)
        nhist : int
            ヒストグラムの分割数
        split : str
            エネルギーファイルの区切りに使っている文字
        hmin : float
            エネルギー領域の最小値
        hmax : float
            エネルギー領域の最大値
        f_out : str
            ヒストグラムデータを記録するファイル名
        probability : bool
            確率分布に規格化するかどうか
        plot : bool
            ヒストグラムを描画するかどうか
        """
        nrep = self.nrep
        data = [[] for irep in range(nrep)]

        # エネルギーの瞬間値を取得
        name, ext = os.path.splitext(f_in)
        for irep in range(nrep):
            with open("{:s}{:02d}{:s}".format(name, irep + 1, ext), "r") as f:
                for line in f:
                    data[irep] += [float(line.replace("\n", "")
                                         .split(split)[column])]

        # 各種変数の取得
        if hmax is None:
            hmax = max([max(data_irep) for data_irep in data])
        if hmin is None:
            hmin = min([min(data_irep) for data_irep in data])
        dh = (hmax - hmin) / nhist
        ham = np.linspace(hmin, hmax, nhist, endpoint=False) + 0.5 * dh

        print("hmax =", hmax)
        print("hmin =", hmin)
        print("dh   =", dh)

        # ヒストグラムの作成
        hist = [np.histogram(data_irep, bins=nhist, range=(hmin, hmax),
                density=probability)[0] for data_irep in data]

        # ファイルに出力
        if f_out is not None:
            name, ext = os.path.splitext(f_out)
            for irep, hist_irep in enumerate(hist):
                with open("{:s}{:02d}{:s}"
                          .format(name, irep + 1, ext), "w+") as f:
                    for iham, ihist in zip(ham, list(hist_irep)):
                        f.write("{:g} {:g}\n".format(iham, ihist))

        # 描画
        if plot:
            plt.xlabel("E")
            if probability:
                plt.ylabel("P(E)")
            else:
                plt.ylabel("H(E)")
            for hist_irep in hist:
                plt.plot(ham, hist_irep)
            plt.show()

    def wham(self, f_in, split=" ", nmax=1000, fname_dos="lndos.dat",
             fname_free="free_energy.dat", plot=True):
        u"""
        WHAMを用いて各レプリカのヒストグラムから状態密度を計算する.

        Parameters
        ----------
        f_in : str
            エネルギーヒストグラムのファイル名(連番数字は抜いて与える)
            1列目にエネルギー、2列目にヒストグラムのみのファイルにしておくこと!
        split : str
            データの区切りに使う文字
        nmax : int
            逐次計算の回数
        fname_dos : str
            状態密度出力ファイル名
        fname_free : str
            自由エネルギーの軌跡の出力ファイル名
        plot : bool
            描画するかどうか
        """
        nrep = self.nrep

        # ヒストグラムデータの取得
        hist = [[] for irep in range(nrep)]
        ham = []
        name, ext = os.path.splitext(f_in)
        for irep in range(nrep):
            with open("{:s}{:02d}{:s}".format(name, irep + 1, ext), "r") as f:
                for line in f:
                    num_list = [float(s) for s in
                                line.replace("\n", "").split(split)]
                    if irep == 0:
                        ham += [num_list[0]]
                    hist[irep] += [num_list[1]]
        hist = np.array(hist)
        ham = np.array(ham)

        # 各変数の取得
        ndata = hist.sum(axis=1).reshape(nrep, 1)
        nhist = hist.shape[1]
        ham_shape = ham.reshape(1, nhist)
        beta = self.beta.reshape(nrep, 1)

        # 自由エネルギーの初期化
        free_energy = np.zeros(nrep)
        free_list = [[free_irep] for free_irep in list(free_energy)]

        # 逐次計算開始 #######################################################
        for istep in range(1, nmax + 1):

            # 各エネルギー帯でのヒストグラムの和の対数の計算
            hist_sum = hist.sum(axis=0)
            hist_sum[hist_sum == 0.0] = -np.inf
            hist_sum[hist_sum > 0.0] = np.log(hist_sum[hist_sum > 0.0])

            # 状態密度の対数を計算する
            term_dos = np.log(ndata) - (beta * ham_shape)\
                + free_energy.reshape(nrep, 1)
            max_dos = term_dos.max(axis=0).reshape(1, nhist)
            lndos = hist_sum - max_dos.reshape(nhist)\
                - np.log(np.sum(np.exp(term_dos - max_dos), axis=0))

            # 自由エネルギーを計算する
            term_f = lndos.reshape(1, nhist) - (beta * ham_shape)
            max_f = term_f.max(axis=1).reshape(nrep, 1)
            free_energy = -max_f.reshape(nrep)\
                - np.log(np.sum(np.exp(term_f - max_f), axis=1))

            # 自由エネルギーの軌跡を保存
            for irep, free_irep in enumerate(free_energy):
                free_list[irep] += [free_irep]

        # 逐次計算終了 #######################################################

        # 結果の記録
        with open(fname_dos, "w+") as f:
            for iham, ilndos in zip(list(ham), list(lndos)):
                f.write("{:g} {:g}\n".format(iham, ilndos))
        with open(fname_free, "w+") as f:
            for istep, ifree in enumerate(np.array(free_list).T):
                tmp = " ".join(map(str, ifree))
                f.write("{:d} {:s}\n".format(istep, tmp))

        # 自由エネルギーの軌跡と状態密度の対数を描画
        if plot:
            fig, (axl, axr) = plt.subplots(ncols=2, figsize=(10, 4))
            axl.set_xlabel("step")
            axl.set_ylabel("free energy")
            for free_list_irep in free_list:
                axl.plot(range(nmax + 1), free_list_irep)
            axr.set_xlabel("E")
            axr.set_ylabel("lndos(E)")
            axr.plot(ham, lndos)
            plt.show()

    def get_average(self, f_in, split=" ", phys=lambda x: x,
                    ntemp=1000, temp_max=None, temp_min=None,
                    f_out="average.dat", plot=True):
        u"""
        状態密度から、物理量の平均値と温度の関係を求める.

        Parameters
        ----------
        f_in : str
            状態密度のデータファイル名
            1列目にエネルギー、2列目に状態密度の対数のみにしておくこと!
        split : str
            データの区切りに使う文字
        phys : function
            計算する物理量(エネルギーの関数)
        ntemp : int
            温度の刻み幅
        temp_max : float
            温度領域の最大値(指定するなら与える)
        temp_min : float
            温度領域の最小値(指定するなら与える)
        f_out : str
            結果の出力ファイル(いらないならNone)
        plot : bool
            描画するかどうか

        Returns
        -------
        (temp, phys_data) : (array(float), array(float))
            温度と平均値の配列
        """
        if temp_max is None:
            temp_max = self.temp.max()
        if temp_min is None:
            temp_min = self.temp.min()
        temp_list = np.linspace(temp_min, temp_max, ntemp, endpoint=True)
        beta_list = 1.0 / (self.k_boltz * temp_list)

        # 状態密度の取得
        lndos = []
        ham = []
        with open(f_in, "r") as f:
            for line in f:
                num_list = [float(s) for s in
                            line.replace("\n", "").split(split)]
                ham += [num_list[0]]
                lndos += [num_list[1]]
        ham = np.array(ham)
        lndos = np.array(lndos)

        # 物理量が0以下になるのを防ぐための補正定数の計算
        fix_num = np.abs(phys(ham).min()) + 1.0

        phys_data = []
        for ibeta in list(beta_list):  # 温度ループ

            # 規格化定数の対数の計算
            term_norm = lndos - ibeta * ham
            max_norm = term_norm.max()
            normal = max_norm + np.log(np.sum(np.exp(term_norm - max_norm)))

            # 平均値の対数の計算
            # 物理量Aが0以下になると対数で処理できないので、 <A+Const> を計算する。
            term_av = np.log(phys(ham) + fix_num) + lndos - ibeta * ham
            max_av = term_av.max()
            average = max_av + np.log(np.sum(np.exp(term_av - max_av)))\
                - normal

            # 平均値を記録( <A> = <A+Const> - Const )
            phys_data += [np.exp(average) - fix_num]

        # 結果をファイルに出力
        if f_out is not None:
            with open(f_out, "w+") as f:
                for i in range(ntemp):
                    f.write("{:g} {:g} {:g}\n".format(temp_list[i],
                            beta_list[i], phys_data[i]))

        # 描画
        if plot:
            plt.xlabel("Temperature")
            plt.plot(temp_list, phys_data)
            plt.show()

        return temp_list, np.array(phys_data)

    def get_specific_heat(self, f_in, split=" ",
                          ntemp=1000, temp_max=None, temp_min=None,
                          f_out="specific_heat.dat", plot=True):
        u"""
        比熱を計算する.

        Parameters
        ----------
        f_in : str
            状態密度のデータファイル名
            1列目にエネルギー、2列目に状態密度の対数のみにしておくこと!
        split : str
            データの区切りに使う文字
        ntemp : int
            温度の刻み幅
        temp_max : float
            温度領域の最大値(指定するなら与える)
        temp_min : float
            温度領域の最小値(指定するなら与える)
        f_out : str
            結果の出力ファイル(いらないならNone)
        plot : bool
            描画するかどうか
        """
        temp, ham_av = self.get_average(f_in=f_in, split=split,
                                        phys=lambda x: x,
                                        ntemp=ntemp, temp_max=temp_max,
                                        temp_min=temp_min,
                                        f_out=None, plot=None)

        ham2_av = self.get_average(f_in=f_in, split=split,
                                   phys=lambda x: x**2,
                                   ntemp=ntemp, temp_max=temp_max,
                                   temp_min=temp_min, f_out=None, plot=None)[1]

        beta = 1.0 / (self.k_boltz * temp)
        heat = (ham2_av - ham_av**2) / (self.k_boltz * temp**2)

        if f_out is not None:
            with open(f_out, "w+") as f:
                for i in range(ntemp):
                    f.write("{:g} {:g} {:g}\n"
                            .format(temp[i], beta[i], heat[i]))

        if plot:
            plt.xlabel("Temperature")
            plt.ylabel("Specific heat")
            plt.plot(temp, heat)
            plt.show()

if __name__ == "__main__":
    rem = REM(fname="temp_sample.out", k_boltz=8.3144598e-3)
    rem.remake_file("energy.dat", "trajectory.dat")
    rem.remake_file("torsion.dat", "trajectory.dat")
    # print(rem.get_roundtrip_rate("trajectory.dat"))
    rem.get_multi_histogram(f_in="energy_temp.dat", column=2, nhist=100,
                            hmax=80, hmin=-120)
    rem.wham(f_in="hist_rem.dat")
    hoge = rem.get_average(f_in="lndos.dat", f_out="pot_curve_wham.dat",
                           phys=lambda x: (x))
    rem.get_specific_heat(f_in="lndos.dat")
