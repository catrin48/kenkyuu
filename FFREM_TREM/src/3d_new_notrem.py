u"""
Last Update:2022/07/18

    多重レベルGHMC法のシミュレーションプログラム→HMC法へ(出力される文字はまだGHMCのままのところがある)
    solv_modelというパラメータによって、3D-RISMまたはGBSAを切り替え可能

    ・実行コマンド
        python <このプログラムのパス> <インプットファイル名>

    Note
    ----
        ・adpはReferenceで回すのが最速
"""

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
from multiprocessing import Pool, Manager
import warnings
import math
import random
import onlypotential_copy
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
kB=8.3144598e-3
# 'Non-optimal GB parameters detected for GB model GBn2' という警告メッセージを無視
warnings.filterwarnings("ignore", message="Non-optimal GB parameters detected for GB model GBn2")

directory_path="."
#file_path=os.path.join(f"energy_ffrem_{temperature}.dat")
def update_history(history_file, final_state):
    with open(history_file, 'a') as f:
        f.write(f"{final_state}\n")
def load_initial_state(history_file):
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            lines = f.readlines()
            # 最後の状態を取得し、リストに変換
            if lines:
                last_state = lines[-1].strip()
                current_state = eval(last_state)  # 文字列をリストに変換
                return current_state
    return [1, 2, 3, 4, 5, 6]  # 

def main(replica_index,value):
    #file_path=os.path.join(f"energy_ffrem_{temperature}.dat")
    uf = funcs.UnitFactor()
    start = time.time()
    restart_switch="ON"
    restart_step=0
    nghmcs=1
    platform_name="Reference"
    solv_model="RISM"
    solv_md_name="GBn2"
    gbsa_md_name="ACE"
    solv_name="GBn2"
    gbsa_name="ACE"
    
    prmtop_MD_name="adp_ff14SBonlysc.prmtop"
    prmtop_name="adp_ff14SBonlysc.prmtop"
    inpcrd_name=f"adp{replica_index+1}.inpcrd"
    init_v_name="Non"
    temperature=value
    timestep=8.0
    nmds=10
    nrespa=8
    #value =torsion,4,6,8,14 torsion,6,8,14,16
    """react_atom_list = []
    for value_i in value:
       react_atom = value_i.split(",")
       react_atom_list += [[react_atom[0]] + list(map(int, react_atom[1:]))]
    """
    react_atom_list=[['torsion', 4, 6, 8, 14], ['torsion', 6, 8, 14, 16]]
    phys_skip=1
    config_skip=         1
    input_rism1d_name=   "input_rism1d"
    input_rism3d_name=   "input_rism3d"
    config_name=         f"adp_configuration_rism_{inpcrd_name}.trj"
    energy_name=         f"adp_energy_rism_{inpcrd_name}.dat"
    react_name=          f"adp_react_{inpcrd_name}.dat"
    react_temp_name=f"adp_react_{temperature}.dat"
    restart_r_name=      f"adp{replica_index+1}.inpcrd"
    restart_v_name=      "restart_v.out"
    energy_temp_name=  f"adp_energy_{temperature}.dat"  
    energy_ffrem_name=f"energy_ffrem_{temperature}.dat"
    file_path=os.path.join(f"energy_ffrem_{temperature}.dat")
    #print("Setting systems ...")
    # インプットファイルからパラメータを取得
   # with open(sys.argv[1], "r") as file:
        #restart_switch          = file.readline().rstrip().split()[1]
        #restart_step            = int(file.readline().rstrip().split()[1])
        #nghmcs                  = int(file.readline().rstrip().split()[1])
        #platform_name           = file.readline().rstrip().split()[1]
        #solv_model              = file.readline().rstrip().split()[1]
        #solv_md_name            = file.readline().rstrip().split()[1]
        #gbsa_md_name            = file.readline().rstrip().split()[1]
        #solv_name               = file.readline().rstrip().split()[1]
        #gbsa_name               = file.readline().rstrip().split()[1]
        #prmtop_MD_name          = file.readline().rstrip().split()[1]
        #prmtop_name             = file.readline().rstrip().split()[1]
        #inpcrd_name             = file.readline().rstrip().split()[1]
        #init_v_name             = file.readline().rstrip().split()[1]
        #temperature             = float(file.readline().rstrip().split()[1])
        ##timestep                = float(file.readline().rstrip().split()[1])
       # noise                   = float(file.readline().rstrip().split()[1])
        #nmds                    = int(file.readline().rstrip().split()[1])
        #nrespa                  = int(file.readline().rstrip().split()[1])

        # 反応座標リストの取得
        #value = file.readline().rstrip().split()[1:]
        #react_atom_list = []
        #for value_i in value:
         #   react_atom = value_i.split(",")
          #  react_atom_list += [[react_atom[0]] + list(map(int, react_atom[1:]))]
        
        #phys_skip               = int(file.readline().rstrip().split()[1])
        #config_skip             = int(file.readline().rstrip().split()[1]) 
        #input_rism1d_name       = file.readline().rstrip().split()[1]
        ##input_rism3d_name       = file.readline().rstrip().split()[1]
        #config_name             = file.readline().rstrip().split()[1]
        #energy_name             = file.readline().rstrip().split()[1]
        #react_name              = file.readline().rstrip().split()[1]
        #restart_r_name          = file.readline().rstrip().split()[1]
        #restart_v_name          = file.readline().rstrip().split()[1]
        #replica_id              = int(file.readline().rstrip().split()[1])  
    """with open(sys.argb[1],"r") as file:
       temperature             = float(file.readline().rstrip().split()[1])
       
       value = file.readline().rstrip().split()[1:]
       react_atom_list = []
       for value_i in value:
            react_atom = value_i.split(",")
            react_atom_list += [[react_atom[0]] + list(map(int, react_atom[1:]))]
       replica_id              = int(file.readline().rstrip().split()[1])"""
    # パラメータの出力
   # print("\n# Parameter ######################################")
   # print(f"Restart step                = {restart_step:12d}")
   # print(f"HMC step                   = {nghmcs:12d}")
   # print(f"Large time step             = {timestep:f} [fs]")
   # print(f"Small time step             = {timestep / nrespa:f} [fs]")
    #print(f"Noise                       = {noise:f}")
   # print(f"MD step                     = {nmds:12d}")
   # print(f"phys_skip                   = {phys_skip:12d}")
   # print(f"config_skip                 = {config_skip:12d}")
    #print("#####################################################\n")

    #print("\n# System Settings ######################################")
   # print(f"Platform                      : {platform_name:s}")
   # print(f"FF for MD calculation         : {prmtop_MD_name:s}")
   # print(f"GBSAModel for MD calculation  : {solv_md_name:s} / {gbsa_md_name:s}")
   # if solv_model == "RISM":
    #    print("Potential for metropolice method is calculated by 3D-RISM")
    #elif solv_model == "GBSA":
     #    print("#")
     #   print(f"FF for metropolice            : {prmtop_name:s}")
      #  print(f"GBSAModel for metropolice     : {solv_name:s} / {gbsa_name:s}")
   # else:
    #print("Invalid 'solv_model': Use 'RISM' or 'GBSA'.")
     #   sys.exit()
    #print("#####################################################\n")
    
    # システムの設定
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

    # ファイルの設定
    config_file = open(config_name, "a+")
    #energy_file = open(energy_name, "a+")
    energy_temp_file=open(energy_temp_name,"a+")
    #energy_ffrem_file=open(energy_ffrem_name,"w+")
    react_file  = open(react_name, "a+")
    config_old_file = open("adp_configration_old.trj", "w+")
    config_new_file = open("adp_configration_new.trj", "w+")
    #acc_file = open("acceptance_temperature.dat", "w+") 
    #acc_name = f"acceptance_{temperature}.dat"
    #with open(acc_name, "w+") as acc_file:

    energy_old_file   = open("energy_old.dat", "w+")
    energy_old_file.write(f"# pe_gbsa_old       pe_rism_old      ke_old\n")

    energy_new_file   = open("energy_new.dat", "w+")
    energy_new_file.write(f"# pe_gbsa_new       pe_rism_new      ke_new\n")

    other_energy_old_file = open("other_enregy_old.dat", "w+")
    other_energy_old_file.write(f"# myu_gbsa_old        myu_rism_old        pe_none_old       myu_delta_old\n")
    
    other_energy_new_file = open("other_enregy_new.dat", "w+")
    other_energy_new_file.write(f"# myu_gbsa_new        myu_rism_new        pe_none_new       myu_delta_new\n")

    time_setting = time.time() - start
    #print(f"Setting elapse time = {time_setting:f} [sec]")
    start_ghmc = time.time()
    #print("Calculating now ...")

    sys.stdout.flush()
    sys.stderr.flush()

    # 1D-RISM計算
    #if solv_model == "RISM":
      #  funcs.get_rism1d(temperature, input_rism1d_name)
        #sb.run("rm -f time*", shell=True)
    
    # 初期座標を取得し、MD計算のインテグレータに渡す
    x_old = simulation_ref.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(uf.length)
    simulation_MD.context.setPositions(Quantity(x_old, uf.length))

    # 初期座標のポテンシャルを計算
    if solv_model == "RISM":
       file_path=os.path.join(f"energy_ffrem_{temperature}.dat")
       if os.path.exists(file_path):
        with open(file_path, 'r') as f:
         pe_old = float(f.readline().strip())
        # print(pe_old)    
        #output_name = funcs.calc_rism_energy(comment, inpcrd_name, prmtop_name, input_rism1d_name, input_rism3d_name, x_old, temperature)
        #pe_rism, myu_rism = funcs.get_rism_energy(output_name)
        #pe_old1 = pe_rism
        #with open(file_path,'w') as f:
         #        f.write(f"{pe_old1}")
        #print("=")
        #print(pe_old1) 
   # U_RISM = U_inter + μ_RISM
        #output_name         = funcs.calc_rism_energy(comment, inpcrd_name, prmtop_name, input_rism1d_name, input_rism3d_name, x_old, temperature)
        #pe_rism, myu_rism   = funcs.get_rism_energy(output_name)
        #pe_old = pe_rism
       # file_path=os.path.join(f"energy_ffrem_{temperature}.dat")
       # with open(file_path,'r') as f:
       else:
        output_name = funcs.calc_rism_energy(comment, inpcrd_name, prmtop_name, input_rism1d_name, input_rism3d_name, x_old, temperature)
        pe_rism, myu_rism = funcs.get_rism_energy(output_name)
        pe_old = pe_rism  
        #print(pe_old) 
        with open(file_path,'w') as f:
         f.write(f"{pe_old}")
   # pe_old = float(f.readline().strip())
        #print("---")
        #print(pe_old*CV_CAL)
     #   print(myu_rism*CV_CAL)
    elif solv_model == "GBSA":
        pe_old = simulation_ref.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(uf.energy)
       # print(f"number1={pe_old:f}")
        simulation_none.context.setPositions(Quantity(x_old, uf.length))
        simulation_none.step(1)
        pe_none   = integrator_none.getGlobalVariableByName("pe_none")
        #print(pe_none)
        #print(pe_old-pe_none)
    ### シミュレーション(start) ###
    energy_ffrem_file=open(energy_ffrem_name,"w+")
   # print("---")
    for ighmcs in range(restart_step + 1, restart_step + nghmcs + 1):

        # モーメンタムリフレッシュ
        simulation_ref.step(1)
        
        # oldのエネルギーを取得し、メトロポリス判定用のインテグレータに値を渡す
        ke_old  = integrator_ref.getGlobalVariableByName("ke_old")
        #print(ke_old)
        E_old   =  ke_old + pe_old
        #print(E_old*CV_CAL)
        integrator_metro.setGlobalVariableByName("E_old", E_old)

        # MD計算のインテグレータに速度を渡す　
        v_old = simulation_ref.context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(uf.speed)
        simulation_MD.context.setVelocities(Quantity(v_old, uf.speed))

        # oldの配位,エネルギーを保存 ---------------------------------------------------------
        position_old    = copy.deepcopy(x_old)
        # データ記録
        if ighmcs % config_skip == 0:
            for r_i in position_old:
                config_old_file.write(" ".join(list(map(str, r_i))) + "\n")
        
        # U_GBSA = U_inter + μ_GBSA / μ_GBSA = U_GBSA - U_inter
        if solv_model == "RISM":
         simulation_gbsa.context.setPositions(Quantity(x_old, uf.length))
         simulation_gbsa.step(1)
         pe_gbsa = integrator_gbsa.getGlobalVariableByName("pe_gbsa")
         simulation_none.context.setPositions(Quantity(x_old, uf.length))
         simulation_none.step(1)
         pe_none   = integrator_none.getGlobalVariableByName("pe_none")
         #print(pe_old*CV_CAL)
         #print(pe_gbsa*CV_CAL) 
         #print(pe_none*CV_CAL)
       #  output_name         = funcs.calc_rism_energy(comment, inpcrd_name, prmtop_name, input_rism1d_name, input_rism3d_name, x_old)
       #  pe_new, myu_rism   = funcs.get_rism_energy(output_name)
         #print(myu_rism*CV_CAL)
         myu_gbsa  = pe_gbsa - pe_none#pe_gbsa:3d-rismの座標でのgbsaの相互作用　pe_none:力場ff14SBでの粒子間の相互作用 myu_gbsa:gbsaの化学ポテンシャル
         myu_rism  = pe_old - pe_none#pe_old:3d-rismでの相互作用　pe_none:ff14SB.での粒子間の相互作用　myu_rism:rismの化学ポテンシャル
         #print(myu_rism*CV_CAL)
         myu_delta = myu_rism - myu_gbsa#myu_delta　同一座標による3d rismとgbsaの化学ポテンシャルのエネルギー差
         energy_old_file.write(f"{pe_gbsa*CV_CAL:f} {pe_old*CV_CAL:f} {ke_old*CV_CAL:f}\n") #RISMの場合　pe_gbsaはrismの座標でのpe_gbsa pe_oldは3d-rismの相互作用
         other_energy_old_file.write(f"{myu_gbsa*CV_CAL:f} {myu_rism*CV_CAL:f} {pe_none*CV_CAL:f} {myu_delta*CV_CAL:f}\n")
        # ---------------------------------------------------------------------------------

        # 1HMCステップ状態を更新
        start_time = time.time()
        simulation_MD.step(1)
        end_time = time.time()
        time_md += end_time - start_time

        x_new = simulation_MD.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(uf.length)
        v_new = simulation_MD.context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(uf.speed)

        # メトロポリス判定用のインテグレータにパラメータを渡す
        ke_new = integrator_MD.getGlobalVariableByName("ke_new")
        integrator_metro.setGlobalVariableByName("ke_new", ke_new)
        # 3D-RISM計算
        if solv_model == "RISM":
            # U_RISM = U_inter + μ_RISM
            output_name         = funcs.calc_rism_energy(comment, inpcrd_name, prmtop_name, input_rism1d_name, input_rism3d_name, x_new, temperature)
            pe_new, myu_rism   = funcs.get_rism_energy(output_name)
            integrator_metro.setGlobalVariableByName("pe_new", pe_new)
            #print(pe_new)            
            # newの配位,エネルギーを保存 -------------------------------------------------------------
            position_new    = copy.deepcopy(x_new)
            # データ記録
            if ighmcs % config_skip == 0:
                for r_i in position_new:
                    config_new_file.write(" ".join(list(map(str, r_i))) + "\n")
        
            # U_GBSA = U_inter + μ_GBSA / μ_GBSA = U_GBSA - U_inter
            simulation_gbsa.context.setPositions(Quantity(x_new, uf.length))
            simulation_gbsa.step(1)
            pe_gbsa = integrator_gbsa.getGlobalVariableByName("pe_gbsa")
            simulation_none.context.setPositions(Quantity(x_new, uf.length))
            simulation_none.step(1)
            pe_none   = integrator_none.getGlobalVariableByName("pe_none")
            #print(pe_gbsa*CV_CAL)
            myu_gbsa  = pe_gbsa - pe_none
            myu_delta = myu_rism - myu_gbsa
            
            energy_new_file.write(f"{pe_gbsa*CV_CAL:f} {pe_new*CV_CAL:f} {ke_new*CV_CAL:f}\n")
            other_energy_new_file.write(f"{myu_gbsa*CV_CAL:f} {myu_rism*CV_CAL:f} {pe_none*CV_CAL:f} {myu_delta*CV_CAL:f}\n")
            # ---------------------------------------------------------------------------------
        elif solv_model == "GBSA":
            simulation_metro.context.setPositions(Quantity(x_new, uf.length))
            
            
        # メトロポリス判定
        simulation_metro.step(1)
        acceptance  = integrator_metro.getGlobalVariableByName("accept")
        
        if acceptance == 0: 
            # reject
            simulation_MD.context.setPositions(Quantity(x_old, uf.length))
            #print(v_old)
            #print("=====")
            simulation_ref.context.setVelocities(Quantity(v_old, uf.speed))
           # print(0)
            #print(pe_old)
            if ighmcs % phys_skip == 0:
                position    = copy.deepcopy(x_old)
                kinetic     = ke_old
                potential   = pe_old

            if ighmcs % nghmcs == 0:
                position    = simulation_ref.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)
         
                velocity    = copy.deepcopy(v_old)
        else:
            # accept       
            simulation_MD.context.setPositions(Quantity(x_new, uf.length))
            simulation_ref.context.setVelocities(Quantity(v_new, uf.speed))
           # print(1)    
            x_old  = copy.deepcopy(x_new)
            if solv_model == "RISM":
                pe_old = pe_new
                #print(pe_new)
                with open(file_path,'w') as f:
                 f.write(f"{pe_new}")
            elif solv_model == "GBSA":
                pe_new = integrator_metro.getGlobalVariableByName("pe_new")
                #print(pe_new*CV_CAL)
                pe_old = pe_new
                #simulation_none.context.setPositions(Quantity(x_old, uf.length))
                #simulation_none.step(1)
                #pe_none   = integrator_none.getGlobalVariableByName("pe_none")
        #print(pe_none)
                #print(pe_old-pe_none)
            if ighmcs % phys_skip == 0:
                position    = copy.deepcopy(x_new)
                kinetic     = integrator_metro.getGlobalVariableByName("ke_new")
                potential   = pe_new

            if ighmcs % nghmcs == 0:
                position    = simulation_MD.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)
                velocity    = copy.deepcopy(v_new)

        # データ記録
        if ighmcs % config_skip == 0:
            for r_i in position:
                config_file.write(" ".join(list(map(str, r_i))) + "\n")

        if ighmcs % phys_skip == 0:
            
            file_name = f"acceptance_{temperature}.dat"
            with open(file_name, "w+") as acc_file:
             acc_file.write(f"{acceptance:g}\n")

            temp_av     += 2.0*kinetic/(3.0*natom*funcs.K_BOLTZ)
            
            energy_temp_file.write(f"{ighmcs:d} {kinetic*CV_CAL:f} {potential*CV_CAL:f} {(kinetic + potential)*CV_CAL:f} \n")
            energy_ffrem_file.write(f"{potential:f}")
            #energy_file.write(f"{ighmcs:d} {kinetic*CV_CAL:f} {potential*CV_CAL:f} {(kinetic + potential)*CV_CAL:f} \n")
            # 反応座標の計算、記録
            react_file.write(f"{ighmcs:d}")
            for react_atom in react_atom_list:
                react = funcs.get_react(position, react_atom[0], react_atom[1:])
                react_file.write(f" {react:f}")
            react_file.write("\n")
            
            #print(f"{ighmcs:d} {kinetic*CV_CAL:f} {potential*CV_CAL:f} {(kinetic + potential)*CV_CAL:f} ")
    ### シミュレーション(end)   ###

    # 結果の出力
    temp_av     /= nghmcs/phys_skip
    accept_mc   = integrator_metro.getGlobalVariableByName("naccept") / nghmcs
    time_ghmc   = time.time() - start_ghmc
    time_total  = time.time() - start
   # print(potential)
   # print("-------------")
    result=potential
   # number2=potential      = {noise:f}")
    #print(f"number{temperature}={potential:f}")
    #print("# Result ############################################")
   # print(f"Calculation time of MD      = {time_md:f} [sec]")
   # print(f"Calculation time            = {time_ghmc:f} [sec]")
   # print(f"Total elapse time           = {time_total:f} [sec]")
   # print(f"Temperature                 = {temp_av:f} [K]")
   # print(f"Acceptance ratio            = {accept_mc*100:f} [%]")
   # print("#####################################################")
  # print(f"{ighmcs:d} {kinetic*CV_CAL:f} {potential*CV_CAL:f} {(kinetic + potential)*CV_CAL:f} ")
    
    step_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    time_1step = time_total/nghmcs
    for step in step_list:
        time_if = time_1step*step
       # print(f"{step:g} step / {time_if:g} [sec] {time_if/60/60:g} [h]")

    # リスタートデータの記録 restart_switchがONのときだけ記録
    if restart_switch == "ON":
        with open(inpcrd_name, "r") as file:
            comment = file.readline().rstrip()
        funcs.save_restart(velocity, position, restart_r_name, restart_v_name, natom, comment, uf)

    energy_temp_file.close()
    config_file.close()
    #energy_file.close()
    react_file.close()
    config_new_file.close()
    acc_file.close()
    return (replica_index,result)

def check_pair(pair, results, directory_path, shared_state, lock):
    index1, index2 = pair
    num1 = results.get(index1)
    num2 = results.get(index2)
    #print(index1,index2)
    #print(replicas[index1],replicas[index2])
    file_name1=f"adp{index1+1}.inpcrd"
    file_name2=f"adp{index2+1}.inpcrd"
    temperature1=replicas[index1]
    temperature2=replicas[index2]
    #print(file_name1,file_name2)
    # num3 = onlypotential.calculation_3d(file_name,temperature)
    #num4 = onlypotential.calculation_3d(file_name,temperarure)
    
    #with concurrent.futures.ThreadPoolExecutor() as executor:
     #   future_num3 = executor.submit(onlypotential_copy.calculate_3d, file_name1, temperature2)
      #  future_num4 = executor.submit(onlypotential_copy.calculate_3d, file_name2, temperature1)
        
        # 計算結果を取得
       # num3 = future_num3.result()
        #num4 = future_num4.result()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_num3 = executor.submit(onlypotential_copy.calculate_3d, file_name1, temperature2)
        future_num4 = executor.submit(onlypotential_copy.calculate_3d, file_name2, temperature1)
        
        # 計算結果を取得
        num3 = future_num3.result()
        num4 = future_num4.result()

    #num3=onlypotential_copy.calculate_3d(file_name1,temperature2)
    #num4=onlypotential_copy.calculate_3d(file_name2,temperature1)
   # print(file_name1,file_name2,temperature1,temperature2,num1,num2,num3,num4)
    if num1 is not None and num2 is not None:
        #diff=(1/kB/replicas[index2]*(num3-num1))+(1/kB/replicas[index1]*(num4-num2))
        diff=(1/kB/temperature1*(num1-num4))+(1/kB/temperature2*(num2-num3))
       # print(diff)
        corrected_diff = math.exp(diff)
        threshold = random.random()

        if corrected_diff>threshold:
            #座標fileの交換を行なっている
            file1 = os.path.join(directory_path, f"adp{index1+1}.inpcrd")
            file2 = os.path.join(directory_path, f"adp{index2+1}.inpcrd")
            temp_file = os.path.join(directory_path, f"temp_swap_{index1}_{index2}.inpcrd")
            #エネルギーfileについて。交換の場合エネルギーが異なるのでremove()でfileを削除している
            file3=os.path.join(directory_path,f"energy_ffrem_{replicas[index1]}.dat")
            file4=os.path.join(directory_path,f"energy_ffrem_{replicas[index2]}.dat")
            #temp_ffrem_file=os.path.join(directory_path,f"temp_swap_{replicas[index1]}_{replicas[index2]}")
            # ファイルの交換
            os.rename(file1, temp_file)
            os.rename(file2, file1)
            os.rename(temp_file, file2)
            
            #os.remove(file3)
            #os.remove(file4)
            with open(file3, 'w') as f:
                f.write(f"{num4:f}")
            with open(file4, 'w') as f:
                f.write(f"{num3:f}")
            # ロックして共有状態を更新
            with lock:
                shared_state[index1], shared_state[index2] = shared_state[index2], shared_state[index1]

            return f"Swapped files: adp{index1+1}.inpcrd and adp{index2+1}.inpcrd due to corrected difference: {corrected_diff}"
        else:
            #return f"Corrected difference between replica {index1} and replica {index2} is 1 or less: {corrected_diff} (No swap)"
            return f"Corrected difference between adp{index1+1} and adp{index2+1} is 1 or less: {corrected_diff} (No swap)"

    return "Results for one or both replicas not found."

if __name__ == "__main__":
    start_time=time.time()
    replicas = [300.0, 350.0, 410.0, 480.0, 550.0, 630.0]
    directory_path = "."  # 実際のディレクトリパスを指定してください
    history_file = 'rireki.dat'

    # 履歴ファイルから初期状態を読み込む
    initial_state = load_initial_state(history_file)

    # 共有メモリに初期状態をセット
    with Manager() as manager:
        shared_state = manager.list(initial_state)  # 共有リストとして初期状態を保存
        lock = manager.Lock()  # ロックを生成

        # 並列でmainを実行してレプリカ結果を取得
        with Pool(processes=6) as pool:
            results_list = pool.starmap(main, list(enumerate(replicas)))

        results = dict(results_list)

        # 半分の確率でペアリストを選択
        if random.random() < 0.5:
            pairs = [(0, 1), (2, 3), (4, 5)]
        else:
            pairs = [(1, 2), (3, 4)]

        # ペアごとの比較を並列で実行
        with Pool(processes=len(pairs)) as pool:
            check_results = pool.starmap(check_pair, [(pair, results, directory_path, shared_state, lock) for pair in pairs])

        # 結果を出力
        for result in check_results:
            print(result)

        # 並列処理が終了した後に最終状態を記録
        final_state = list(shared_state)  # 共有リストを通常のリストに変換
        update_history(history_file, final_state)  # 履歴に最終状態を追加
        end_time=time.time()
        print(f"{end_time-start_time}")
