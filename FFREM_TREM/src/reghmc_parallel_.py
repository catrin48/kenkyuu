u"""
レプリカ交換GHMC法の並列化プログラム.

・r-RESPA法とHMR法を使用.
・連番ファイルは"ファイル名nn.拡張子"という形式で作られる(nは数字).
　(連番ファイルは温度ごとではなくレプリカごとに作られるので，各自で温度ごとに整理すること)
注1)レプリカの温度配置を自分で設定するときは関数set_temperatureをうまくいじってください.
注2)リスタートを行う場合は置換がリセットされるので、交換の軌跡を見るときは注意.
注3)現段階では，一つのプロセスに対しレプリカを一つだけ割り当てるようにしているので，
　  一つのプロセスに複数のレプリカを割り当てる場合は改良が必要(2020，4/1).

一つのプロセスに複数レプリカを割り当てるように改良してみよう
"""
from openmm.app import *
from openmm import *
from simtk.unit import *
from sys import stdout
import sys
import os
import numpy as np
import time
from mpi4py import MPI
from itertools import chain
import random    #2025 1/10 変更した。random数を生成するため

# 定数
K_BOLTZ = 8.3144598e-3  # ボルツマン定数 [kJ/K・mol]
PI = np.pi              # 円周率
HMASS = 1.00794         # 水素原子の質量 [Da]
NANO = 10               # [nm] -> [Å] の変換定数


# シミュレーションに必要な入力 ################################################

# パラメータ
restart_step = 0         # 事前に計算したステップ数(リスタートの場合)
nghmcs = 200       # 全GHMCステップ
timestep = 8.0           # MDの時間ステップ [fs]
noise = 0    # モーメンタムリフレッシュのパラメータ
nmds = 10                # 1GHMCステップごとのMDステップ数
nreplica = 6             # レプリカの個数
temp_range = (300, 630)  # 温度範囲 [K]
hmass_rate = 3.0         # 水素原子の質量の変換倍率(HMR法)
ts_rate = (8,)           # 時間ステップの倍率{n}(r-RESPA法)

# 各プロセスに割り当てるレプリカの数
allocate_list = [1] * nreplica

# 二面角の粒子番号
torsion_atom_list = [(4, 6, 8, 14), (6, 8, 14, 16)]

# 連番じゃないファイル名
prmtop_name = "adp.prmtop"
trajec_name = "trajectory.dat"  # レプリカ交換の軌跡の出力ファイル名
temp_name = "temp_sample.out"   # 温度配置の出力ファイル名
pot_av_name = "pot_av.dat"      # ポテンシャル平均値の出力ファイル名

# 連番ファイルの名前
config_skip = 10  # 配位の記録間隔
fname_orig = {}
fname_orig["inpcrd"] = 'init_r.inpcrd'
fname_orig["init_v"] = None  # 指定なしはNone
fname_orig["config"] = "configuration.dat"
fname_orig["energy"] = "energy.dat"
fname_orig["torsion"] = "torsion.dat"
fname_orig["restart_r"] = "init_r.inpcrd"
fname_orig["restart_v"] = "restart_v.out"

###########################################################################


class UnitFactor:
    u"""OpenMMのシステムに渡すときの単位定数."""

    def __init__(self):
        u"""
        単位定数の設定.

        単位系
            長さ       : [nm]
            時間       : [fs]
            質量       : [Da]
            エネルギー  : [kJ/mol]
            温度       : [K]

            長さ、時間、質量の単位とエネルギー、温度の単位は合っていないので注意!
        """
        self.time = femtosecond
        self.length = nanometer
        self.mass = dalton
        self.speed = self.length / self.time
        self.force = self.mass * self.length / self.time**2
        self.energy = kilojoule_per_mole
        self.temp = kelvin



def set_temperature(temp_range, nreplica, fname):
    u"""
    レプリカの温度配置を設定.

    ・比熱が一定の系を仮定して、等比数列で設定する.
    注)相転移を起こす系の場合はうまくいかないので注意!
    """
    temp_min = temp_range[0]
    temp_max = temp_range[1]

    # 自分で一つ得たい温度があるとき、Tminをうまく設定し直す
    target_temp = 360
    target_replica = 1
    tmp = (target_temp / temp_max)**(1.0 / (nreplica - target_replica - 1))
    temp_min = temp_max * tmp**(nreplica - 1)

    alpha = (temp_max / temp_min)**(1 / (nreplica - 1))
    
    temperature = temp_min * alpha**np.arange(nreplica)
    #print(temperature)
    # 温度配置を記録しておく
    with open(fname, "w+") as file:
        for temp_i in temperature:
            file.write(f"{temp_i:g}\n")

    return temperature

def get_torsion(position, atom):
    u"""
    二面角を計算する.

    (r_ij × r_kj) ・ (r_jk × r_lk) = |r_ij × r_kj||r_jk × r_lk|cos(θ)
    """
    i, j, k, l = atom
    rij = position[i] - position[j]
    rkj = position[k] - position[j]
    rjk = position[j] - position[k]
    rlk = position[l] - position[k]
    p1 = np.cross(rij, rkj)
    p2 = np.cross(rjk, rlk)
    cos_theta = np.dot(p1, p2) / np.sqrt(np.dot(p1, p1) * np.dot(p2, p2))
    theta = np.arccos(cos_theta) * np.sign(np.dot(rkj, np.cross(p1, p2)))

    return theta


def set_forcegroup(system):
    u"""
    r-RESPA法における力のグループ分けを設定する.

    ・使用する力場 ff14SB(溶媒効果はGB/SAのHCT)
    ・力の分け方(番号は時間ステップの昇順)
        0 : HarmonicBondForce, HarmonicAngleForce, HarmonicTorsionForce
        1 : NonBondedForce, CustomGBForce
    """
    # 力のグループ分け
    f_list = system.getForces()
    bonded = [f for f in f_list if isinstance(f, HarmonicBondForce)][0]
    angle = [f for f in f_list if isinstance(f, HarmonicAngleForce)][0]
    torsion = [f for f in f_list if isinstance(f, PeriodicTorsionForce)][0]
    nonbonded = [f for f in f_list if isinstance(f, NonbondedForce)][0]
    customgb = [f for f in f_list if isinstance(f, CustomGBForce)][0]
    bonded.setForceGroup(0)
    angle.setForceGroup(0)
    torsion.setForceGroup(0)
    nonbonded.setForceGroup(1)
    customgb.setForceGroup(1)

    return system


def GHMC(temperature, timestep, noise, nmds, ts_rate, uf):
    u"""
    GHMC法のインテグレータの作成.

    ・MD計算部分にr-RESPA法を使用.
    注1)力のグループ数を変えたときはMD計算部分を変えること.
    注2)周期境界条件については全く考慮されてないのでいるときは各自で実装してください.
    """
    integrator = CustomIntegrator(timestep * uf.time)
    k_boltz = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA

    # 変数の宣言
    integrator.addGlobalVariable("NSTEP", 0)
    integrator.addGlobalVariable("accept", 0)
    integrator.addGlobalVariable("naccept", 0)
    integrator.addGlobalVariable("kT", k_boltz * (temperature * uf.temp))
    integrator.addPerDofVariable("sigma", 0)
    integrator.addPerDofVariable("u", 0)
    integrator.addGlobalVariable("phi", noise)
    integrator.addPerDofVariable("x_old", 0)
    integrator.addPerDofVariable("v_old", 0)
    integrator.addGlobalVariable("KE", 0)
    integrator.addGlobalVariable("H", 0)
    integrator.addGlobalVariable("H_old", 0)
    integrator.addGlobalVariable("weight", 0)

    # 各グループの時間ステップを設定
    tmp = np.array((1,) + ts_rate)
    dt_list = timestep * tmp.cumprod() / tmp.prod()
    for i, dt_i in enumerate(dt_list):
        integrator.addGlobalVariable(f"dt{i:d}", dt_i * uf.time)

    # GHMC計算 #########################################################
    integrator.addUpdateContextState()

    # モーメンタムリフレッシュ
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")
    integrator.addComputePerDof("u", "sigma*gaussian")
    integrator.addComputePerDof("v", "u")
    
    # 状態とエネルギーの確保
    integrator.addComputePerDof("x_old", "x")
    integrator.addComputePerDof("v_old", "v")
    integrator.addComputeSum("KE", "0.5*m*v*v")
    integrator.addComputeGlobal("H_old", "KE+energy")

    # MD計算で状態を更新(r-RESPA法、力のグループ数は2)
    for imds_1 in range(1, nmds + 1):
        integrator.addComputePerDof("v", "v+0.5*dt1*f1/m")
        for imds_0 in range(1, ts_rate[0] + 1):
            integrator.addComputePerDof("v", "v+0.5*dt0*f0/m")
            integrator.addComputePerDof("x", "x+dt0*v")
            integrator.addComputePerDof("v", "v+0.5*dt0*f0/m")
        integrator.addComputePerDof("v", "v+0.5*dt1*f1/m")

    # 遷移確率の計算
    integrator.addComputeSum("KE", "0.5*m*v*v")
    integrator.addComputeGlobal("H", "KE+energy")
    integrator.addComputeGlobal("weight", "exp(-(H-H_old)/kT)")

    # 遷移判定とモーメンタムフリップ
    integrator.addComputeGlobal("accept", "step(weight-uniform)")
    integrator.addComputePerDof("x", "x*accept+x_old*(1-accept)")
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")  # sigmaは温度に依存するスケール因子
    integrator.addComputePerDof("u", "sigma*gaussian")
    integrator.addComputePerDof("v", "u")
    integrator.addComputeGlobal("naccept", "naccept+accept")
    integrator.addComputeGlobal("NSTEP", "NSTEP+1")
    ###################################################################

    return integrator


def REM(step, pot_list, permut, beta_list, accept, uf):
    u"""
    レプリカ交換を行う.

    ・交換アルゴリズムはDEO
        隣接レプリカ対(i,i+1)のうち奇数レプリカ(2i-1,2i)を奇数ステップで、
        偶数レプリカ対(2i,2i+1)を偶数ステップで交換する.
    """
    nreplica = beta_list.size

    # 交換を行うペアの早い方のレプリカのインデックスを取得
    if step % 2 == 1:
        idx_list = range(nreplica)[:-1:2]
    else:
        idx_list = range(nreplica)[1:-1:2]

    for idx in idx_list:  # 交換するペアのループ(idxは温度のラベル)

        # 温度 idx，idx + 1 を持つレプリカのラベルを取得
        prev_rep = permut.index(idx)
        next_rep = permut.index(idx + 1)

        # 交換確率の計算
        delta_beta = beta_list[idx + 1] - beta_list[idx]
        delta_pot = pot_list[next_rep] - pot_list[prev_rep]
        weight = np.exp(delta_beta * delta_pot)

        # 交換の採択
        if (weight > 1.0) or (weight > np.random.uniform()):
            accept[idx] += 1.0
            permut[prev_rep], permut[next_rep] = permut[next_rep], permut[prev_rep]

    return permut, accept


def save_restart(position, velocity, fname_r, fname_v, comment, factor):
    u"""
    リスタートファイルの作成.

    ・配位についてはAmberInpcrdファイル形式で書き込む.
    注)粒子の座標の絶対値が10000 [Å] 未満でないとバグるので注意!
    """
    # 速度の書き込み
    with open(fname_v, "w+") as file:
        for v_i in velocity:
            file.write(" ".join(list(map(str, v_i))) + "\n")

    # 配位の書き込み
    r_inpcrd = position * factor
    natom = r_inpcrd.shape[0]
    with open(fname_r, "w+") as file:
        file.write(comment + "\n")
        file.write(f"{natom:>6d}\n")
        for iatom, r_i in enumerate(r_inpcrd):
            for coordinate in r_i:
                file.write(f" {coordinate:>11.5f}")
            if iatom % 2 == 1:
                file.write("\n")


if __name__ == "__main__":

    uf = UnitFactor()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    console = 0

    # レプリカ数，プロセス数，割り当てに矛盾があったらエラー
    if (nreplica != sum(allocate_list)) or (len(allocate_list) != size):
        if rank == console:
            print("Error")
        sys.exit()

    # 自分に割り当てられたレプリカの数やラベルを取得
    num_myrep = allocate_list[rank]
    label_myrep = []
    for i in range(num_myrep):
        label_myrep += [sum(allocate_list[:rank]) + i]
    comm.barrier()
    label_list = comm.gather(label_myrep, root=console)

    # 温度をコンソールから受信する際のタグを設定
    temp_tag = [11 + i for i in range(size)]

    if rank == console:
        start = time.time()
        print("Setting systems ...")

    # 連番ファイル名の作成
    fname = {}
    for key, value in fname_orig.items():
        if value is not None:
            name, ext = os.path.splitext(value)
            fname[key] = [(name + f"{i + 1:02d}" + ext) for i in label_myrep]
        else:
            fname[key] = [None for i in label_myrep]

    # コンソールでレプリカの温度配置を設定して，各レプリカに送信
    comm.barrier()
    if rank == console:
        temp_list = set_temperature(temp_range, nreplica, temp_name)
        #print(temp_list)
        temperature = temp_list[label_myrep]
        # print(temperature)
        for i in range(size):
            if i != console:
                comm.send(temp_list[label_list[i]], dest=i, tag=temp_tag[i])

    # コンソールから温度を受信
    comm.barrier()
    if rank != console:
        temperature = comm.recv(source=console, tag=temp_tag[rank])

    # システムの設定
    prmtop = AmberPrmtopFile(prmtop_name)
    platform = Platform.getPlatformByName('Reference')
    system = prmtop.createSystem(implicitSolvent=HCT, removeCMMotion=False,
                                 hydrogenMass=hmass_rate * HMASS * uf.mass)
    system = set_forcegroup(system)
    inpcrd = [AmberInpcrdFile(name_i) for name_i in fname["inpcrd"]]
    integrator = [GHMC(temp_i, timestep, noise, nmds, ts_rate, uf)
                  for temp_i in temperature]
    simulation = [Simulation(prmtop.topology, system, integ_i, platform)
                  for integ_i in integrator]

    # 変数の設定
    if rank == console:
        accept_rem = np.zeros(nreplica - 1)      # レプリカ交換のアクセプタンス
        pot_av = np.zeros(nreplica)              # ポテンシャルの平均値 [kJ/mol]
        beta_list = 1.0 / (K_BOLTZ * temp_list)  # 逆温度 [mol/kJ]
        natom = system.getNumParticles()         # 粒子数
        permut = list(range(nreplica))           # レプリカ -> 温度の置換
        trajec_file = open(trajec_name, "w+")    # 交換の軌跡(置換)の出力ファイル

    # 連番ファイルの作成
    config_file = [open(name_i, "w+") for name_i in fname["config"]]
    energy_file = [open(name_i, "w+") for name_i in fname["energy"]]
    torsion_file = [open(name_i, "w+") for name_i in fname["torsion"]]

    # 初期配置と初期速度の設定
    for i in range(num_myrep):
        simulation[i].context.setPositions(inpcrd[i].positions)
        velocity = []
        if fname["init_v"][i] is not None:
            with open(fname["init_v"][i], "r") as file:
                for line in file:
                    velocity += [[float(s) for s in line.split()]]
            simulation[i].context.setVelocities(Quantity(velocity, uf.speed))
        else:
              random_seed=random.randint(0,2**31-1)
              simulation[i].context\
                .setVelocitiesToTemperature(temperature[i] * uf.temp, random_seed)

    comm.barrier()
    if rank == console:
        time_setting = time.time() - start
        print(f"Setting elapse time = {time_setting:8.3f} [sec]")
        start_ghmc = time.time()
        print("Calculating now ...\n")

    # シミュレーション ##############################################################
    for ighmcs in range(restart_step + 1, restart_step + nghmcs + 1):

        # 1GHMCステップの計算
        for simulation_i in simulation:
            simulation_i.step(1)
        potential = [simulation_i.context.getState(getEnergy=True)
                     .getPotentialEnergy().value_in_unit(uf.energy)
                     for simulation_i in simulation]

        # コンソールでレプリカ交換を行い、軌跡を記録
        comm.barrier()
        pot_list = comm.gather(potential, root=console)
        if rank == console:
            pot_list = list(chain.from_iterable(pot_list))  # 1列に揃える
            permut, accept_rem =\
                REM(ighmcs, pot_list, permut, beta_list, accept_rem, uf)
            trajec_file.write(f"{ighmcs:d}")
            for permut_i in permut:
                trajec_file.write(f" {permut_i:d}")
            trajec_file.write("\n")

        # 各レプリカに交換後の温度を送信
        if rank == console:
            temperature_old = temperature
            temperature = temp_list[np.array(permut)[label_myrep]]
            for i, label in enumerate(label_list):
                if i != console:
                    comm.send(temp_list[np.array(permut)[label]],
                              dest=i, tag=temp_tag[i])

        # コンソールから温度を受信
        comm.barrier()
        if rank != console:
            temperature_old = temperature
            temperature = comm.recv(source=console, tag=temp_tag[rank])

        # 交換による温度変化をsimulationに反映し，運動量をリスケールする
        for i in range(num_myrep):
            beta_inv_internal = AVOGADRO_CONSTANT_NA * BOLTZMANN_CONSTANT_kB *\
                (temperature[i] * uf.temp)
            integrator[i].setGlobalVariableByName("kT", beta_inv_internal)
            velocity = simulation[i].context.getState(getVelocities=True)\
                .getVelocities(asNumpy=True).value_in_unit(uf.speed)
            velocity *= np.sqrt(temperature[i] / temperature[i])
            simulation[i].context.setVelocities(Quantity(velocity, uf.speed))

        # 温度ごとのポテンシャルの平均値を計算
        if rank == console:
            pot_av[permut] += np.array(pot_list)

        # 物理量の計算と記録 -------------------------------------------------------
        for i in range(num_myrep):
            state = simulation[i].context.getState(getPositions=True,
                                                   getEnergy=True)

            # 配位
            position = state.getPositions(asNumpy=True).value_in_unit(uf.length)
            if ighmcs % config_skip == 0:
                for r_i in position:
                    config_file[i].write(" ".join(list(map(str, r_i))) + "\n")

            # エネルギー
            kinetic = state.getKineticEnergy().value_in_unit(uf.energy)
            hamiltonian = kinetic + potential[i]
            energy_file[i].write(f"{ighmcs:d} {kinetic:g} {potential[i]:g} " +
                                 f"{hamiltonian:g}\n")

            # 二面角
            torsion_file[i].write(f"{ighmcs:d}")
            for torsion_atom in torsion_atom_list:
                torsion = get_torsion(position, torsion_atom)
                torsion_file[i].write(f" {torsion:g}")
            torsion_file[i].write("\n")
        # ------------------------------------------------------------------------
    ###############################################################################

    # 結果の出力
    accept_mc = [integ_i.getGlobalVariableByName("naccept") / nghmcs
                 for integ_i in integrator]
    comm.barrier()
    accept_mc_list = comm.gather(accept_mc, root=console)
    if rank == console:
        accept_mc_list = list(chain.from_iterable(accept_mc_list))  # 1列に揃える
        pot_av /= nghmcs
        accept_rem /= nghmcs / 2
        time_ghmc = time.time() - start_ghmc
        time_total = time.time() - start
        #print("# Parameter ############################################")
        #print(f"Restart step       = {restart_step:8d}")
        #print(f"GHMC step          = {nghmcs:8d}")
        #print(f"Time step          = {timestep:8.3f} [fs]")
        #print(f"Noise              = {noise:8.3f}")
        #print(f"MD step            = {nmds:8d}")
        #print("Hydrogen mass rate =", hmass_rate)
        #print("Time step rate     =", ts_rate)
        #print("########################################################\n")
        print("# Acceptance of GHMC ###################################")
        for i, accept_i in enumerate(accept_mc_list):
            print(f"replica{i + 1:02d} : {accept_i * 100:8.3f} [%]")
        #print("########################################################\n")
        print("# Acceptance of REM ####################################")
        for i, accept_i in enumerate(accept_rem):
            print(f"temperature{i + 1:02d} <=> temperature{i + 2:02d} : " +
                  f"{accept_i * 100:8.3f} [%]")
        #print("########################################################\n")
        print(f"Calculation time  = {time_ghmc:8.3f} [sec]")
        print(f"Total elapse time = {time_total:8.3f} [sec]\n")

    # ポテンシャルと温度の関係を記録
    if rank == console:
        with open(pot_av_name, "w+") as file:
            for temp_i, pot_i in zip(temp_list, pot_av):
                file.write(f"{temp_i:g} {pot_i:g}\n")

    # リスタートデータの記録など(番号が温度の昇順になるようにしておく)
    position = [simulation[i].context.getState(getPositions=True)
                .getPositions(asNumpy=True).value_in_unit(uf.length)
                for i in range(num_myrep)]
    velocity = [simulation[i].context.getState(getVelocities=True)
                .getVelocities(asNumpy=True).value_in_unit(uf.speed)
                for i in range(num_myrep)]
    comm.barrier()
    position_list = comm.gather(position, root=console)
    velocity_list = comm.gather(velocity, root=console)
    if rank == console:
        position_list = list(chain.from_iterable(position_list))
        velocity_list = list(chain.from_iterable(velocity_list))  # 1列に揃える
        for i, permut_i in enumerate(permut):
            name, ext = os.path.splitext(fname_orig["inpcrd"])
            inpcrd_name = name + f"{permut_i + 1:02d}" + ext
            with open(inpcrd_name, "r") as file:
                comment = file.readline().rstrip()
            name, ext = os.path.splitext(fname_orig["restart_r"])
            restart_r_name = name + f"{permut_i + 1:02d}" + ext
            name, ext = os.path.splitext(fname_orig["restart_v"])
            restart_v_name = name + f"{permut_i + 1:02d}" + ext
            save_restart(position_list[i], velocity_list[i], restart_r_name,
                         restart_v_name, comment, NANO)

    if rank == console:
        trajec_file.close()
    for i in range(num_myrep):
        config_file[i].close()
        energy_file[i].close()
        torsion_file[i].close()
