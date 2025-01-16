u"""
2022/04/13.

OpenMMの力場や溶媒環境を設定する処理に関するライブラリ（OpenMM：http://openmm.org）

    *多重レベルGHMC法 3D-RISM*
      integrator_md
        力場        ：Amberのff14SBonlysc
        溶媒モデル   ：None
      integrator_md
        力場        ：Amberのff14SBonlysc
        溶媒モデル   ：GBneck2+SA
      integrator_ref
        力場        ：Amberのff14SBonlysc
        溶媒モデル   ：None

Note
----
・このライブラリの内容は使う力場や溶媒環境の条件に合わせて自分で書き換えてください
"""
from openmm.app import *
from openmm import *
from simtk.unit import *
import functions as funcs


def set_forcegroup(system):
    u"""
    r-RESPA法における力のグループ分けを設定する.

    ・力場　：ff14SBonlysc
    ・力の分け方
        小さな時間ステップで計算(f0)：HarmonicBondForce, HarmonicAngleForce, HarmonicTorsionForce
        大きな時間ステップで計算(f1)：NonBondedForce, CustomGBForce
    """
    f_list      = system.getForces()
    bonded      = [f for f in f_list if isinstance(f, HarmonicBondForce)][0]
    angle       = [f for f in f_list if isinstance(f, HarmonicAngleForce)][0]
    torsion     = [f for f in f_list if isinstance(f, PeriodicTorsionForce)][0]
    nonbonded   = [f for f in f_list if isinstance(f, NonbondedForce)][0]
    customgb    = [f for f in f_list if isinstance(f, CustomGBForce)][0]
    bonded.setForceGroup(0)
    angle.setForceGroup(0)
    torsion.setForceGroup(0)
    nonbonded.setForceGroup(1)
    customgb.setForceGroup(1)

def set_system_MD(solv_name, gbsa_name, prmtop, temperature, uf, method):
    u"""
        GHMC計算用のシステム.
        OpenMMのcreateSystemクラスを使って系の力場に関する設定を行う.

        使用している設定
        ・GHMC法  ・・・溶質の誘電率：1.0 溶媒の誘電率：78.5 HMR法及びr-RESPA法を適用
    """

    # インプットファイルによって溶媒モデルを設定する
    if solv_name == "HCT":
        solv = HCT
    elif solv_name == "GBn2":
        solv = GBn2

    if gbsa_name == "ACE":
        sa = "ACE"
    elif gbsa_name == "None":
        sa = None
    
    if method == "GHMC":
        system = prmtop.createSystem(
            implicitSolvent=solv,
            removeCMMotion=False,
            temperature=temperature * uf.temp,
            hydrogenMass=3.0 * funcs.HMASS * uf.mass,  # HMR法
            soluteDielectric=1.0,
            solventDielectric=78.5,
            gbsaModel=sa
        )
        set_forcegroup(system)  # r-RESPA法

    return system

def set_system(solv_name, gbsa_name, prmtop, temperature, uf, method):
    u"""
        最終的な精度におけるシステム.
    """

    # インプットファイルによって溶媒モデルを設定する
    if solv_name == "HCT":
        solv = HCT
    elif solv_name == "GBn2":
        solv = GBn2

    if gbsa_name == "ACE":
        sa = "ACE"
    elif gbsa_name == "None":
        sa = None
    
    if method == "GHMC":
        system = prmtop.createSystem(
            implicitSolvent=solv,
            removeCMMotion=False,
            temperature=temperature * uf.temp,
            hydrogenMass=3.0 * funcs.HMASS * uf.mass,  # HMR法
            soluteDielectric=1.0,
            solventDielectric=78.5,
            gbsaModel=sa
        )
        set_forcegroup(system)  # r-RESPA法

    return system