u"""
2021/06/14.

    OpenMMを利用するシミュレーションに使う関数ライブラリ（OpenMM：http://openmm.org）
 ghmcではなくhmcになっている　3D-RISMとGBSAでレプリカ交換で詳細釣り合い満たすため
"""
from openmm.app import *
from openmm import *
from simtk.unit import *

def momentumRefreshmentIntegrator(temperature, timestep, uf):
    u"""
        モーメンタムリフレッシュメント
    """
    integrator = CustomIntegrator(timestep * uf.time)
    kb_internal = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA

    integrator.addGlobalVariable("ke_old", 0)
    integrator.addGlobalVariable("pe_old", 0)

    # 変数の宣言
    integrator.addGlobalVariable("kT", kb_internal * (temperature * uf.temp))
    integrator.addPerDofVariable("sigma", 0)
    integrator.addPerDofVariable("u", 0)
    #integrator.addGlobalVariable("phi", 0)
    #integrator.addGlobalVariable("phi",noise)

    # モーメンタムリフレッシュ
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")
    integrator.addComputePerDof("u", "sigma*gaussian")
    integrator.addComputePerDof("v", "u"             )
    #integrator.addComputePerDof("v", "v*sin(phi)+u*cos(phi)")
    #integrator.addComputePerDof("v", "v*cos(phi)+u*sin(phi)")
    integrator.addComputeSum("ke_old", "0.5*m*v*v")
    integrator.addComputeGlobal("kT","kT")

    return integrator

# インテグレータ
def GHMCIntegrator(timestep, nsteps, nrespa, uf):
    u"""
        GHMC法のインテグレータ.

        ・OpenMMToolsのHMCIntegratorを参考にした。（https://openmmtools.readthedocs.io）
        ・MD計算部分にr-RESPA法を使用。
        ・速度および座標に一切の拘束条件を課さないことを仮定している。（createSystemのremoveCMmotionをFalseにする）
    """
    integrator = CustomIntegrator(timestep * uf.time)

    # 変数の宣言
    integrator.addGlobalVariable("ntrials", 0)
    integrator.addGlobalVariable("ke_new", 0)

    # 各グループの時間ステップを設定
    integrator.addGlobalVariable("dt_rapid", (timestep / nrespa) * uf.time)
    integrator.addGlobalVariable("dt_slow", timestep * uf.time)

    # GHMC計算 #####################################################################################
    integrator.addUpdateContextState()

    # MD計算で状態を更新(r-RESPA法)
    for _ in range(nsteps):
        integrator.addComputePerDof("v", "v+0.5*dt_slow*f1/m")
        for _ in range(nrespa):
            integrator.addComputePerDof("v", "v+0.5*dt_rapid*f0/m")
            integrator.addComputePerDof("x", "x+dt_rapid*v")
            integrator.addComputePerDof("v", "v+0.5*dt_rapid*f0/m")
        integrator.addComputePerDof("v", "v+0.5*dt_slow*f1/m")
    ###############################################################################################

    integrator.addComputeSum("ke_new", "0.5*m*v*v")

    integrator.addComputeGlobal("ntrials", "ntrials+1")

    return integrator

def metropoliceIntegrator(temperature, timestep, uf, solv_model):
    integrator = CustomIntegrator(timestep * uf.time)

    kb_internal = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA

    integrator.addGlobalVariable("kT", kb_internal * (temperature * uf.temp))

    integrator.addGlobalVariable("E_old", 0)
    integrator.addGlobalVariable("ke_new", 0)
    integrator.addGlobalVariable("pe_new", 0)
    integrator.addGlobalVariable("E_new", 0)

    integrator.addGlobalVariable("weight", 0)
    integrator.addGlobalVariable("accept", 0)
    integrator.addGlobalVariable("naccept", 0)
    
    if solv_model == "GBSA":
        integrator.addComputeGlobal("pe_new", "energy")
    integrator.addComputeGlobal("E_new", "ke_new+pe_new")
        
    # 遷移判定とモーメンタムフリップ
    integrator.addComputeGlobal("weight", "exp(-(E_new-E_old)/kT)")
    integrator.addComputeGlobal("accept", "step(weight-uniform)")

    integrator.addComputeGlobal("naccept", "naccept+accept")
    
    
    return integrator


def GBSAPotentialIntegrator(timestep, uf):
    integrator = CustomIntegrator(timestep * uf.time)

    integrator.addGlobalVariable("pe_gbsa", 0)

    integrator.addComputeGlobal("pe_gbsa", "energy")

    return integrator

def NonePotentialIntegrator(timestep, uf):
    integrator = CustomIntegrator(timestep * uf.time)

    integrator.addGlobalVariable("pe_none", 0)

    integrator.addComputeGlobal("pe_none", "energy")

    return integrator
