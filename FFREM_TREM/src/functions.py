u"""
2021/06/14.

自作の定数や関数の定義
"""
import os
import sys
import copy
import time
import numpy as np
from openmm.app import *
from openmm import *
from simtk.unit import *
import subprocess as sb

# 定数定義
K_BOLTZ     = 8.3144598e-3      # ボルツマン定数 [kJ/K・mol]
HMASS       = 1.00794           # 水素原子の質量 [Da]


class UnitFactor:
    u"""OpenMMのシステムに渡すときの単位定数."""

    def __init__(self):
        u"""
        単位定数の設定.

        単位系
            時間　　　　： [fs]
            長さ　　  　： [nm]
            質量　　　　： [Da]
            エネルギー　： [kJ/mol]
            温度　　　　： [K]

            長さ、時間、質量の単位とエネルギー、温度の単位は合っていないので注意!
        """
        self.time   = femtosecond
        self.length = nanometer
        self.mass   = dalton
        self.speed  = self.length / self.time
        self.force  = self.mass * self.length / self.time**2
        self.energy = kilojoule_per_mole
        self.temp   = kelvin


def get_react(position, name, atom):
    u"""
    生体分子の反応座標を計算する.

    ・nameで反応座標の種類を指定し，atomで粒子の番号を指定する
    ・今のところは二面角と原子間距離の計算が実装されている
    """
    if name == "torsion":  # 二面角
        if len(atom) != 4:
            print("Invarid atom list!")
            sys.exit()
        i, j, k, ll = atom
        rij = position[i] - position[j]
        rkj = position[k] - position[j]
        rjk = position[j] - position[k]
        rlk = position[ll] - position[k]
        p1 = np.cross(rij, rkj)
        p2 = np.cross(rjk, rlk)
        cos_theta = np.dot(p1, p2) / np.sqrt(np.dot(p1, p1) * np.dot(p2, p2))
        react = np.arccos(cos_theta) * np.sign(np.dot(rkj, np.cross(p1, p2)))

    elif name == "distance":  # 原子間距離
        if len(atom) != 2:
            print("Invarid atom list!")
            sys.exit()
        i, j = atom
        react = np.sqrt(((position[i] - position[j])**2).sum())

    else:
        print("Invarid name of reaction coordinate!")
        sys.exit()

    return react


def set_initial(simulation, inpcrd, fname_v, temperature, uf):
    u"""
    系の初期状態を設定（OpenMMの機能を利用）.

    ・fname_vが存在しないときはマクスウェル分布から初期速度を設定
    """
    simulation.context.setPositions(inpcrd.positions)

    # 速度
    if os.path.isfile(fname_v):
        print(f"Initial velocities are set by '{fname_v:s}'.")
        velocity = []
        with open(fname_v, "r") as file:
            for line in file:
                velocity += [[float(s) for s in line.split()]]
        simulation.context.setVelocities(Quantity(velocity, uf.speed))
    else:
        #print("Initial velocities are set by Maxwell's distribution.")
        simulation.context.setVelocitiesToTemperature(temperature * uf.temp, 1124)


def save_restart(velocity, position, fname_r, fname_v, natom, comment, uf):
    u"""
    リスタートファイルの作成.

    ・配位についてはAmberInpcrdファイル形式で書き込む.
    ・ [Å] で書き込むことに注意する.
    ・粒子の座標の絶対値が10000 [Å] 未満でないと多分バグるので注意!
    """
    
    # 速度の書き込み
    with open(fname_v, "w+") as file:
        for v_i in velocity:
            file.write(" ".join(list(map(str, v_i))) + "\n")

    # 配位の書き込み
    with open(fname_r, "w+") as file:
        file.write(comment + "\n")
        file.write(f"{natom:>6d}\n")
        for iatom, r_i in enumerate(position):
            for coordinate in r_i:
                file.write(f" {coordinate:>11.5f}")
            if iatom % 2 == 1:
                file.write("\n")

def get_rism1d(temperature, input_rism1d_name):
    u'''
        溶媒の感受率を計算する1D-RISMを実行する関数
        xvvファイルのファイル名（文字列）を返す
    '''

    temp_name = f"_{int(temperature):d}"

    rism1d_name_tmp = input_rism1d_name + temp_name

    start_time = time.time()
    sb.run(f"cp {input_rism1d_name:s}.in {rism1d_name_tmp:s}.inp", shell=True)
    sb.run(f"rism1d {rism1d_name_tmp:s} > /dev/null", shell=True)

    sb.run(f"cp {rism1d_name_tmp:s}.xvv {input_rism1d_name:s}.xvv", shell=True)

    # 不要な一時ファイルの削除
    sb.run(f"rm -f {rism1d_name_tmp:s}* time*", shell=True)

def update_rism1d_name(temperature, input_rism1d_name):
    return f"{input_rism1d_name}{temperature}" 


def get_inptraj(file_name, position, factor, comment=None):
    u'''
        inptrajファイルを作成する関数
        position（1step分）を書き込む.
    '''

    file = open(file_name, "w+")

    # 一番初めにコメントを加える
    file.write(comment + "\n")

    # inpcrd形式で座標を書き込む
    position *= factor
    cnt = 0
    for position_i in position:
        for coordinate in position_i:
            cnt += 1
            file.write(f"{coordinate:>8.3f}")
            if cnt % 10 == 0:
                file.write("\n")
    if cnt % 10 != 0:
        file.write("\n")
    
    file.close()

def calc_rism_energy(comment, inpcrd_name, prmtop_name, input_rism1d_name, input_rism3d_name, position, temperature):
    u'''
    22/06/16.

        3D-RISMに基づいたポテンシャルエネルギーを計算する関数.
        sanderを利用した1点計算をしている(imin=5,maxcyc=1でポテンシャルのみ抽出).

        毎ステップ呼び出しているので、その分の計算コストがかかっている.
            2022/06~ Pythonのマルチプロセスで並行計算による時間短縮
            2022/06/23 post-processing計算は、inptrajに書き込んだステップをnstepとして、(nstep-1)点のポテンシャルを得られる
            2022/06/30 ２回計算の無駄を削除→マルチプロセスの削除
    '''

    # 単位変換の定数
    CV_ANG  = 10.0            # [nm]  -> [Å]

    # 出力ファイル名
    output_name     = f"sander_{temperature}.out"
    position_copy   = copy.deepcopy(position)
    inptraj_name    = f"rism_inptraj_{temperature}.inpcrd"

    time_file   = open(f"time_sander.dat", "a")

    # 座標をAmber対応の形式(inpcrd)に変換
    get_inptraj(inptraj_name, position_copy, CV_ANG, comment)
    
    input_rism1d_name_ = update_rism1d_name(temperature, input_rism1d_name)
    ### ----- 3D-RISM計算 ----- ###
    start_time = time.time()    
    cmd = f"sander -O -i {input_rism3d_name:s}.in -o {output_name:s} -p {prmtop_name:s} -c {inpcrd_name:s} -y {inptraj_name:s} -xvv {input_rism1d_name_:s}.xvv"
    sb.run(cmd, shell=True)
    end_time = time.time()
    time_sander = end_time - start_time

    time_file.write(f'{time_sander:g}\n')

    # 邪魔なファイルの削除
    #cmd = f"rm -f {inptraj_name:s}"
    #sb.run(cmd, shell=True)

    return output_name

def get_rism_energy(output_name):
    u'''
    22/06/23.
    
        outputファイルから、ポテンシャルエネルギーを読み取る関数.
    '''

    CV_J = 4.184           # [cal] -> [J]
    line = ""

    file = open(output_name, "r")

    while("FINAL RESULTS" not in line):
        line = file.readline()
    for i in range(5):
        line = file.readline()
    potential = float(line.rstrip().split()[1]) * CV_J
    for i in range(3):
        line = file.readline()
    pe_rism = float(line.rstrip().split()[8]) * CV_J

    # 邪魔なファイルの削除
    #cmd = f"rm -f {output_name:s} restrt"
    #sb.run(cmd, shell=True)

    return potential, pe_rism
