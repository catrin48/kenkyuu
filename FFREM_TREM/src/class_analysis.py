u"""
解析に関するプログラムをまとめてある.

DataAnalysisTools
    物理量を解析
ConfigurationAnalysisTools
    座標データの解析
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DataAnalysisTools:
    u"""解析ツールを一括管理."""

    def __init__(self, fname, split=None, eqstep=0):
        u"""
        ファイルからデータを読み込み、列ごとにnumpy配列datasetに格納しておく.

        Parameter
        ---------
        fname : str
            データファイル名
        split : str
            データの区切りに使われている文字
        eqstep : int
            平衡化で捨てるデータ数
        """
        self.eqstep = eqstep
        self.dataname = fname

        self.dataset = []
        with open(fname, "r") as f:
            for line in f:
                if line != "\n":
                    self.dataset += [[float(s) for s in
                                      line.replace("\n", "").split(split)]]
        self.dataset = np.array(self.dataset).T[:, eqstep:]

    def skip_data(self, fname=None, nskip=10):
        u"""
        データを間引いて出力し直す.

        Parameter
        ---------
        fname : str
            出力ファイル名(デフォルト有り)
        nskip : int
            間引き数
        """
        if fname is None:
            name, ext = os.path.splitext(self.dataname)
            fname = name + "_skip" + ext

        dataset_skip = self.dataset[:, ::nskip]
        with open(fname, "w+") as f:
            for line in dataset_skip.T:
                f.write("{:s}\n".format(" ".join(map(str, line))))

    def get_histogram_1d(self, column, nhist=100, xmax=None, xmin=None,
                         probability=False, fname="hist1d.dat", plot=True):
        u"""
        1つのデータに対してヒストグラムを作成する.

        Parameter
        ---------
        column : int
            入力ファイルの何列目のデータを使うか(0から数える)
        nhist : int
            ヒストグラムの分割数
        xmax : float
            ヒストグラムの範囲の最大値(固定したいときは指定する)
        xmin : float
            ヒストグラムの範囲の最小値(固定したいときは指定する)
        probability : bool
            確率密度にするかどうか
        fname : str
            出力ファイル名(いらないときはNoneを与える)
        plot : bool
            グラフを描画するかどうか
        """
        data = self.dataset[column]
        if xmax is None:
            xmax = data.max()
        if xmin is None:
            xmin = data.min()
        dx = (xmax - xmin) / nhist

        # ヒストグラムの計算
        hist, xedge = np.histogram(data, bins=nhist, range=(xmin, xmax),
                                   density=probability)
        x = np.delete(xedge, -1) + 0.5 * dx  # 範囲の中心を代表点にする

        if fname is not None:
            with open(fname, "w+") as f:
                for ix, ihist in zip(x, hist):
                    f.write("{:g} {:g}\n".format(ix, ihist))

        if plot:
            plt.xlabel("x")
            if probability:
                plt.ylabel("P(x)")
            else:
                plt.ylabel("H(x)")
            plt.plot(x, hist)
            plt.show()

    def get_histogram_2d(self, column, nhist=(20, 20),
                         xmax=None, xmin=None, ymax=None, ymin=None,
                         probability=False, fname="hist2d.dat", plot=True):
        u"""
        2つのデータに対してヒストグラムを作成する.

        Parameter
        ---------
        column : (int, int)
            入力ファイルの何列目のデータを使うか(0から数える)
            x軸にcolumn[0]、y軸にcolumn[1]が入る
        nhist : (int, int)
            ヒストグラムの分割数
            nhist[0]がx軸の分割数、nhist[1]がy軸の分割数
        xmax : float
            ヒストグラムの範囲のx軸の最大値(固定したいときは指定する)
        xmin : float
            ヒストグラムの範囲のx軸の最小値(固定したいときは指定する)
        ymax : float
            ヒストグラムの範囲のy軸の最大値(固定したいときは指定する)
        ymin : float
            ヒストグラムの範囲のy軸の最小値(固定したいときは指定する)
        probability : bool
            確率密度にするかどうか
        fname : str
            出力ファイル名(いらないときはNoneを与える)
        plot : bool
            グラフを描画するかどうか
        """
        data_x, data_y = self.dataset[column[0]], self.dataset[column[1]]
        if xmax is None:
            xmax = data_x.max()
        if xmin is None:
            xmin = data_x.min()
        if ymax is None:
            ymax = data_y.max()
        if ymin is None:
            ymin = data_y.min()
        dx = (xmax - xmin) / nhist[0]
        dy = (ymax - ymin) / nhist[1]

        # ヒストグラムの計算
        hist, xedge, yedge = np.histogram2d(data_x, data_y, bins=nhist,
                                            range=((xmin, xmax), (ymin, ymax)),
                                            density=probability)
        x = np.delete(xedge, -1) + 0.5 * dx
        y = np.delete(yedge, -1) + 0.5 * dy

        if fname is not None:
            with open(fname, "w+") as f:
                for ix in range(nhist[0]):
                    for iy in range(nhist[1]):
                        f.write("{:g} {:g} {:g}\n".format(x[ix], y[iy],
                                hist[ix][iy]))
                    f.write("\n")

        if plot:
            X, Y = np.meshgrid(x, y)
            ax = Axes3D(plt.figure())
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if probability:
                ax.set_zlabel("P(x, y)")
            else:
                ax.set_zlabel("H(x, y)")
            ax.plot_wireframe(X, Y, hist)
            plt.show()

    def box_average(self, column, fname="error.dat", plot=True):
        u"""
        Box Averageを用いてデータの統計誤差を計算する.

        Parameter
        ---------
        column : int
            入力ファイルの何列目のデータを使うか(0から数える)
        fname : str
            分散の軌跡の出力ファイル名(いらないときはNoneを与える)
        plot : bool
            描画するかどうか

        Returns
        -------
        error_list : list(float)
            誤差(標準偏差)のnumpy配列
        """
        data = self.dataset[column]
        ndata = data.size

        # 分散と"分散の誤差"
        var = data.var() / (ndata - 1.0)
        var_err = var * (2.0 / (ndata - 1.0))**0.5
        var_list = [var]
        var_err_list = [var_err]

        step = 0
        while ndata >= 10:  # Block averageのループ(データ数が10になったら切る)
            step += 1

            # ndataが奇数のときは、端のデータを捨てる
            if ndata % 2 == 1:
                data = np.delete(data, -1)

            # 2つまとまりで平均を取り、新たなデータを作る
            data = 0.5 * (data[::2] + data[1::2])
            ndata = data.size

            # 新たな分散と"分散の誤差"の計算
            var = data.var() / (ndata - 1.0)
            var_err = var * (2.0 / (ndata - 1.0))**0.5
            var_list += [var]
            var_err_list += [var_err]

        # 分散の軌跡をファイルに記録
        if fname is not None:
            with open(fname, "w") as f:
                for istep in range(step + 1):
                    f.write("{:d} {:g} {:g}\n".format(istep, var_list[istep],
                                                      var_err_list[istep]))

        # 分散の軌跡を描画
        if plot:
            plt.title("variance")
            plt.xlabel("step")
            plt.errorbar(range(step + 1), var_list, yerr=var_err_list, fmt="o")
            plt.show()

        return np.sqrt(np.array(var_list))

    def get_cumulative_average(self, fname="cumulative_av.dat",
                               column_step=0, plot=True):
        u"""
        データの累積平均を計算して記録する.

        Parameter
        ---------
        fname : str
            出力ファイル名(いらなければNone)
        column_step : int
            ステップデータの列(なければNone)
        plot : bool
            描画するかどうか
        """
        nv, nstep = self.dataset.shape
        step = np.arange(1, nstep + 1)
        stepdata = np.arange(self.eqstep + 1, nstep + self.eqstep + 1)

        # データセットの中にステップデータがある場合は削除しておく
        dataset = self.dataset
        if column_step is not None:
            dataset = np.delete(dataset, column_step, axis=0)

        # 累積平均の計算
        cumulative_av = dataset.cumsum(axis=1) / step

        if fname is not None:
            with open(fname, "w+") as f:
                for istep, icum in zip(stepdata, cumulative_av.T):
                    tmp = " ".join(map(str, icum))
                    f.write("{:d} {:s}\n".format(istep, tmp))

        if plot:
            plt.xlabel("step")
            for idx, icum in zip(stepdata, cumulative_av):
                plt.ylabel("cumulative_av[{:d}]".format(idx + 1))
                plt.plot(step, icum)
                plt.show()

    def get_correlation_time(self, column, tmax=50, fname="correlation.dat",
                             plot=True):
        u"""
        データの相関時間を計算する.

        Parameter
        ---------
        column : int
            データの列
        tmax : int
            tmax**2ステップまでの相関を調べる
        fname : str
            データの出力ファイル名
        plot : bool
            プロットするか
        Returns : list of float
            データのリスト
        """
        data = self.dataset[column]
        ndata = data.size
        data_var = data.var()

        # 異常な場合はnanを返す
        if data_var == 0.0:
            return [np.nan]
        # tmaxが大きすぎたら警告して終了
        if tmax**2 > ndata:
            print("tmax is too large!")
            return [np.nan]

        cor_time_list = []
        t_list = np.arange(1, tmax + 1)
        for tstep in t_list**2:
            tmp = ndata % tstep
            if tmp == 0:
                block_av = np.mean(data.reshape((-1, tstep)), axis=1)
            else:
                # 余ったデータは切り捨てる
                block_av = np.mean(data[:-tmp].reshape((-1, tstep)), axis=1)
            cor_time = tstep * block_av.var() / data_var
            cor_time_list += [cor_time]

        # データの記録と描画
        if fname is not None:
            with open(fname, "w+") as f:
                for tstep_sqrt, cor_time in zip(t_list, cor_time_list):
                    f.write("{:g} {:g}\n".format(tstep_sqrt, cor_time))

        if plot:
            plt.xlabel("Step^(1/2)")
            plt.ylabel("Correlation Time")
            plt.plot(np.array(t_list), cor_time_list, "o")
            plt.show()

        return cor_time_list


class ConfigurationAnalysisTools:
    u"""座標データの解析ツール."""

    def __init__(self, fname, natom, split=" ", eqstep=0):
        u"""
        座標データを読み込む.

        Parameter
        ---------
        fname : str
            座標データのファイル名
        natom : int
            系の粒子数
        split : str
            データの区切りに使われている文字
        eqstep : int
            平衡化で捨てるステップ数
        """
        self.natom = natom

        # 座標データを各ステップごとに配列に保存
        self.data = []
        config = []
        with open(fname, "r") as file:
            for count, line in enumerate(file):
                config += [[float(s) for s in
                            line.replace("\n", "").split(split)]]

                # 1ステップ分のデータの配列をself.dataに保存
                if (count + 1) % self.natom == 0:
                    self.data += [config]
                    config = []

        self.data = np.array(self.data)[eqstep:]

    def get_torsion(self, i, j, k, l, unit="degree", fname=None):
        u"""
        i-j-k-lで繋がった4粒子の二面角Θ [rad] を計算する.

        (r_ij × r_kj) ・ (r_jk × r_lk) = |r_ij × r_kj||r_jk × r_lk|cos(θ)

        Parameter
        ---------
        i, j, k, l : int
            粒子の番号を指定するインデックス
        unit : str
            角度の単位 "degree":度数法, "radian":弧度法
        fname : str
            出力ファイル名(Noneは出力なし)

        Returns
        -------
        torsion_data : ndarray
            二面角の瞬間値リスト
        """
        torsion_data = []

        for r in list(self.data):

            # 二面角の計算
            rij = r[i] - r[j]
            rkj = r[k] - r[j]
            rjk = r[j] - r[k]
            rlk = r[l] - r[k]
            p1 = np.cross(rij, rkj)
            p2 = np.cross(rjk, rlk)
            cos_theta = np.dot(p1, p2)\
                / np.sqrt(np.dot(p1, p1) * np.dot(p2, p2))
            theta = np.arccos(cos_theta)\
                * np.sign(np.dot(rkj, np.cross(p1, p2)))

            # 記録
            if unit == "degree":
                torsion_data += [np.rad2deg(theta)]
            else:
                torsion_data += [theta]

        if fname is not None:
            with open(fname, "w+") as file:
                for istep, torsion in enumerate(torsion_data):
                    file.write("{:d} {:g}\n".format(istep + 1, torsion))

        return np.array(torsion_data)

    def save_analysis_data(self, dataset, fname):
        u"""
        解析データをファイルに出力.

        Parameter
        ---------
        dataset : list
            (データの種類の数)×(ステップ数)の形で書き込みたい瞬間値データを与える
        fname : str
            出力ファイル名
        """
        dataset = np.array(dataset)
        with open(fname, "w+") as file:
            for istep, data_istep in enumerate(dataset.T):
                file.write("{:d} ".format(istep + 1))
                file.write("{:s}\n"
                           .format(" ".join(list(map(str, data_istep)))))


if __name__ == "__main__":
    """
    energy = DataAnalysisTools("energy_temp03.dat")
    torsion =\
        DataAnalysisTools("torsion_temp03.dat")
    energy.skip_data(nskip=20)
    torsion.skip_data(nskip=20)
    # energy.get_histogram_1d(column=2, nhist=200, fname=None,
    #                         probability=True)
    torsion.get_histogram_2d(column=(2, 1), nhist=(100, 100),
                             fname="torsionhist.dat", probability=True,
                             xmax=np.pi, xmin=-np.pi, ymax=np.pi, ymin=-np.pi)
    # hoge = energy.get_correlation_time(column=2, tmax=50, fname=None)
    # energy.get_cumulative_average(fname=None)
    """
    hoge = DataAnalysisTools("torsionhist.dat")
    print(hoge.dataset[2].max())
