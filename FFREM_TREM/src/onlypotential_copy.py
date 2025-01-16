# 2024/5/12 このファイルはonlypotentia.pyを大分変更してある、更に変更するつまりはある。
import os
import sys
import time
import numpy as np
from openmm.app import *
from openmm import *
from simtk.unit import *
import functions as funcs
import methods
import forcefields
import subprocess as sb
from multiprocessing import Pool
import os
import warnings
import math
import random
# 'Non-optimal GB parameters detected for GB model GBn2' という警告メッセージを無視
warnings.filterwarnings("ignore", message="Non-optimal GB parameters detected for GB model GBn2")

#temperatures = [300, 400, 500, 600, 700, 800]

def calculate_gb(file_name,temperature):
    uf = funcs.UnitFactor()
    start = time.time()

    platform_name           = "Reference"
    solv_model              = "GBSA"
    solv_md_name            = "GBn2"
    gbsa_md_name           = "ACE"
    solv_name               = "GBn2"
    gbsa_name               = "ACE"
    prmtop_MD_name          = "adp_ff14SBonlysc.prmtop"
    prmtop_name             = "adp_ff14SBonlysc.prmtop"
    inpcrd_name             = file_name
    timestep=8.0
    nmds=10
    nrespa=8    
    temperature             = temperature
    input_rism1d_name       = f"input_rism1d{temperature}.xvv"
    input_rism3d_name       = "input_rism3d"
    init_v_name="Non"
    #    config_name             = file.readline().rstrip().split()[1]
   #     energy_name             = file.readline().rstrip().split()[1]
  #      react_name              = file.readline().rstrip().split()[1]
 #       restart_r_name          = file.readline().rstrip().split()[1]
#        restart_v_name          = file.readline().rstrip().split()[1]

    inpcrd          = AmberInpcrdFile(inpcrd_name)
    prmtop_MD       = AmberPrmtopFile(prmtop_MD_name)
    prmtop          = AmberPrmtopFile(prmtop_name)
    platform        = Platform.getPlatformByName(platform_name)

    #  モーメンタムリフレッシュ部分のインテグレータの設定
    system_ref       = forcefields.set_system(solv_name, gbsa_name, prmtop, temperature, uf, method="GHMC")
    integrator_ref   = methods.momentumRefreshmentIntegrator(temperature, timestep, uf)
    simulation_ref   = Simulation(prmtop.topology, system_ref, integrator_ref, platform)
    funcs.set_initial(simulation_ref, inpcrd, init_v_name, temperature, uf)

    # md部分のインテグレータの設定
    system_MD      = forcefields.set_system_MD(solv_md_name, gbsa_md_name, prmtop_MD, temperature, uf, method="GHMC")
    integrator_MD  = methods.GHMCIntegrator(timestep, nmds, nrespa, uf)
    simulation_MD  = Simulation(prmtop_MD.topology, system_MD, integrator_MD, platform)

    # メトロポリス判定部分のインテグレータの設定
    system_metro       = forcefields.set_system(solv_name, gbsa_name, prmtop, temperature, uf, method="GHMC")
    integrator_metro   = methods.metropoliceIntegrator(temperature, timestep, uf, solv_model)
    simulation_metro   = Simulation(prmtop.topology, system_metro, integrator_metro, platform)

    # 内部エネルギーのみを計算するインテグレータ

    # インプットファイルによって溶媒モデルを設定する
    if solv_name == "HCT":
        solv = HCT
    elif solv_name == "GBn2":
        solv = GBn2

    if gbsa_name == "ACE":
        sa = "ACE"
    elif gbsa_name == "None":
        sa = None

    system_gbsa = prmtop.createSystem(
            implicitSolvent=solv,
            removeCMMotion=False,
            temperature=temperature * uf.temp,
            hydrogenMass=3.0 * funcs.HMASS * uf.mass,  # HMR法
            soluteDielectric=1.0,
            solventDielectric=78.5,
            gbsaModel=sa
        )
    system_none = prmtop.createSystem(
        implicitSolvent=None,
        removeCMMotion=False,
        soluteDielectric=1.0,
        solventDielectric=78.5,
    )
    integrator_gbsa   = methods.GBSAPotentialIntegrator(timestep, uf)
    simulation_gbsa   = Simulation(prmtop.topology, system_gbsa, integrator_gbsa, platform)
    integrator_none   = methods.NonePotentialIntegrator(timestep, uf)
    simulation_none   = Simulation(prmtop.topology, system_none, integrator_none, platform)

    # 変数の設定
    CV_CAL      = 1.0/4.184          # [J] -> [cal]
    temp_av     = 0.0
    natom       = system_MD.getNumParticles()
    time_md     = 0.0   # 時間発展部分だけの計算時間
    with open(inpcrd_name, "r") as file:
        comment = file.readline().rstrip()


    
    sys.stdout.flush()
    sys.stderr.flush()

    # 1D-RISM計算
    if solv_model == "RISM":
        funcs.get_rism1d(temperature, input_rism1d_name)
        sb.run("rm -f time*", shell=True)

                  
    x_old = simulation_ref.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(uf.length)
    simulation_MD.context.setPositions(Quantity(x_old, uf.length))

    # 初期座標のポテンシャルを計算
    if solv_model == "RISM":
        # U_RISM = U_inter + μ_RISM
        output_name         = funcs.calc_rism_energy(comment, inpcrd_name, prmtop_name, input_rism1d_name, input_rism3d_name, x_old)
        pe_rism, myu_rism   = funcs.get_rism_energy(output_name)
        pe_old = pe_rism
        #print(f"number3={pe_old*CV_CAL:f}")
    elif solv_model == "GBSA":
        pe_old = simulation_ref.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(uf.energy)
        #print(f"number4={pe_old:f}") 
        return pe_old


def calculate_3d(file_name,temperature):
    uf = funcs.UnitFactor()
    start = time.time()

    platform_name           = "Reference"
    solv_model              = "RISM"
    solv_md_name            = "GBn2"
    gbsa_md_name            = "ACE"
    solv_name               = "GBn2"
    gbsa_name               = "ACE"
    prmtop_MD_name          = "adp_ff14SBonlysc.prmtop"
    prmtop_name             = "adp_ff14SBonlysc.prmtop"
    inpcrd_name             = file_name
    temperature             = temperature
    timestep=8.0
    nmds=10
    nrespa=8
    input_rism1d_name       = f"input_rism1d"
    input_rism3d_name       = "input_rism3d"
    init_v_name="non"
    #    config_name             = file.readline().rstrip().split()[1]
   #     energy_name             = file.readline().rstrip().split()[1]
  #      react_name              = file.readline().rstrip().split()[1]
 #       restart_r_name          = file.readline().rstrip().split()[1]
#        restart_v_name          = file.readline().rstrip().split()[1]

    inpcrd          = AmberInpcrdFile(inpcrd_name)
    prmtop_MD       = AmberPrmtopFile(prmtop_MD_name)
    prmtop          = AmberPrmtopFile(prmtop_name)
    platform        = Platform.getPlatformByName(platform_name)

    #  モーメンタムリフレッシュ部分のインテグレータの設定
    system_ref       = forcefields.set_system(solv_name, gbsa_name, prmtop, temperature, uf, method="GHMC")
    integrator_ref   = methods.momentumRefreshmentIntegrator(temperature, timestep, uf)
    simulation_ref   = Simulation(prmtop.topology, system_ref, integrator_ref, platform)
    funcs.set_initial(simulation_ref, inpcrd, init_v_name, temperature, uf)

    # md部分のインテグレータの設定
    system_MD      = forcefields.set_system_MD(solv_md_name, gbsa_md_name, prmtop_MD, temperature, uf, method="GHMC")
    integrator_MD  = methods.GHMCIntegrator(timestep, nmds, nrespa, uf)
    simulation_MD  = Simulation(prmtop_MD.topology, system_MD, integrator_MD, platform)

    # メトロポリス判定部分のインテグレータの設定
    system_metro       = forcefields.set_system(solv_name, gbsa_name, prmtop, temperature, uf, method="GHMC")
    integrator_metro   = methods.metropoliceIntegrator(temperature, timestep, uf, solv_model)
    simulation_metro   = Simulation(prmtop.topology, system_metro, integrator_metro, platform)

    # 内部エネルギーのみを計算するインテグレータ

    # インプットファイルによって溶媒モデルを設定する
    if solv_name == "HCT":
        solv = HCT
    elif solv_name == "GBn2":
        solv = GBn2

    if gbsa_name == "ACE":
        sa = "ACE"
    elif gbsa_name == "None":
        sa = None

    system_gbsa = prmtop.createSystem(
            implicitSolvent=solv,
            removeCMMotion=False,
            temperature=temperature * uf.temp,
            hydrogenMass=3.0 * funcs.HMASS * uf.mass,  # HMR法
            soluteDielectric=1.0,
            solventDielectric=78.5,
            gbsaModel=sa
        )
    system_none = prmtop.createSystem(
        implicitSolvent=None,
        removeCMMotion=False,
        soluteDielectric=1.0,
        solventDielectric=78.5,
    )
    integrator_gbsa   = methods.GBSAPotentialIntegrator(timestep, uf)
    simulation_gbsa   = Simulation(prmtop.topology, system_gbsa, integrator_gbsa, platform)
    integrator_none   = methods.NonePotentialIntegrator(timestep, uf)
    simulation_none   = Simulation(prmtop.topology, system_none, integrator_none, platform)

    # 変数の設定
    CV_CAL      = 1.0/4.184          # [J] -> [cal]
    temp_av     = 0.0
    natom       = system_MD.getNumParticles()
    time_md     = 0.0   # 時間発展部分だけの計算時間
    with open(inpcrd_name, "r") as file:
        comment = file.readline().rstrip()



    sys.stdout.flush()
    sys.stderr.flush()

    # 1D-RISM計算
    #if solv_model == "RISM":
     #   funcs.get_rism1d(temperature, input_rism1d_name)
      #  sb.run("rm -f time*", shell=True)


    x_old = simulation_ref.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(uf.length)
    simulation_MD.context.setPositions(Quantity(x_old, uf.length))

    # 初期座標のポテンシャルを計算
    if solv_model == "RISM":
        # U_RISM = U_inter + μ_RISM
        output_name         = funcs.calc_rism_energy(comment, inpcrd_name, prmtop_name, input_rism1d_name, input_rism3d_name, x_old,temperature)
        pe_rism, myu_rism   = funcs.get_rism_energy(output_name)
        pe_old = pe_rism
        #print(f"number3={pe_old:f}")
        return pe_old
    elif solv_model == "GBSA":
        pe_old = simulation_ref.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(uf.energy)
        #print(f"number4={pe_old:f}")
        return 
def calculate_pair(file_x,file_x_prime,temperature):
 a=calculate_gb(file_x,temperature)
 #b1=calculate_3d(file_x_prime,temperature)
 c=calculate_gb(file_x_prime,temperature)
 d=calculate_3d(file_x,temperature)
 file_path=os.path.join(f"energy_ffrem_{temperature}.dat")
 if os.path.exists(file_path):
  with open(file_path, 'r') as f:
            # ファイルの最初の行を読み取り、float に変換
            b = float(f.readline().strip())
    #except FileNotFoundError:
     #   print(f"Error: File {file_path} not found.")
      #  return None
    #except ValueError:
     #   print(f"Error: Invalid value in {file_path}.")
      #  return None
 else:
   b=calculate_3d(file_x_prime,temperature)
   print("keisansita")
 result=math.exp(-1/(temperature*8.3144598e-3)*((c+d)-(a+b)))
# print(a,b,c,d,result,temperature)
 
 return result,b,d
def process_file_pair(task):
    file_x, file_x_prime, temp = task
    energy_ffrem_name = f"energy_ffrem_{temp}.dat"
    # 通常のペアで計算
    result,b,d = calculate_pair(file_x, file_x_prime, temp)

    # ランダムな閾値を生成して比較
    threshold = random.random()

    # 結果が閾値を超えている場合、ファイルペアを交換
    if result > threshold:
        swapped = True
        temp_file_x = f"{file_x}.temp"  # 一時ファイル名
        #ここは座標file名の交換になっている
        os.rename(file_x, temp_file_x)             # file_x -> temp
        os.rename(file_x_prime, file_x)            # file_x_prime -> file_x
        os.rename(temp_file_x, file_x_prime)       # temp -> file_x_prime
        energy_ffrem_name=f"energy_ffrem_{temp}.dat"
        #アクセプトの場合、エネルギーの値も交換なのでdの値をff_fileに書き込んでいる
        with open(energy_ffrem_name,"w") as energy_ffrem_file:
          energy_ffrem_file.write(f"{d}\n")
             
        return (file_x_prime, file_x, temp, result, swapped)
    else:
        swapped = False
        with open(energy_ffrem_name,"w") as energy_ffrem_file:
         energy_ffrem_file.write(f"{b}\n")
        return (file_x, file_x_prime, temp, result, swapped)

def main():
     start_time=time.time()
     file_pairs = [
        ("init_r01.inpcrd", "adp1.inpcrd"),  # (x, x')
        ("init_r02.inpcrd", "adp2.inpcrd"),  # (y, y')
        ("init_r03.inpcrd", "adp3.inpcrd"),  # (z, z')
        ("init_r04.inpcrd", "adp4.inpcrd"),  # (i, i')
        ("init_r05.inpcrd", "adp5.inpcrd"),  # (j, j')
        ("init_r06.inpcrd", "adp6.inpcrd")   # (k, k')
     ]
     temperatures = [300.0, 350.0, 410.0, 480.0, 550.0, 630.0]  # 温度設定

    # file_pairsとtemperaturesを結合して並列処理用の引数リストを作
     tasks = [(file_pairs[i][0], file_pairs[i][1], temperatures[i]) for i in range(len(file_pairs))]

    # 並列処理を行う
     with Pool() as pool:
        results = pool.map(process_file_pair, tasks)
     
    # 結果を出力
     for result in results:
        file_x, file_x_prime, temp, res, swapped = result
        if swapped:
            print(f"Result ({res}) exceeds random threshold, files swapped: {file_x} and {file_x_prime}")
        else:
            print(f"Files: {file_x} and {file_x_prime}, Temp: {temp}, Result: {res}")
     
     print("Total execution time:", time.time() - start_time)

if __name__ == "__main__":
    main()
