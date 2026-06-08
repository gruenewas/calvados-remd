import calvados as cal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
   # parser.add_argument('--T',nargs='?', default='.', const='.', type=str)
    parser.add_argument('--name',type=str)
    parser.add_argument('--sysname',type=str)
   # parser.add_argument('-sim_time',type=str,default="9")
    parser.add_argument('--path',type=str)
   # parser.add_argument('--replica',type=str,default="1")
    parser.add_argument('--ref_name',type=str)
    args = parser.parse_args()
   #start = {"260.15":int(20e3),"280.15":int(22e3),"290.15":int(20e3),"310.15":int(21e3)}
    name = args.name
    sysname = args.sysname
    folder = f"{args.path}/{sysname}"
    ref_name = args.ref_name
    print(sysname)
    T=float(sysname.split('_')[-1].replace('K',''))
    #chaindict = fix_trajectory(path=folder,simlog_path=f"{folder}/run.log",mixed_pdb="equilibration_final.pdb",mixed_dcd=f"{sysname}.dcd",overwrite=True,start=0,step=20)
    chaindict = {name:(0,99)}
    #plot_e2e_corr_function(path=args.path,sysname=sysname,temp=T,comp_dict=chaindict,out_path=f"{folder}/data")
    client_names = [i for i in chaindict.keys() if i != ref_name]
    client_chain_list = [chaindict[i] for i in client_names]
    slab = cal.analysis.SlabAnalysis(
        name = name,
        input_path = folder,
        input_pdb = f"top.pdb",
        #input_dcd = f"{sysname}_first0.5mus.dcd",
        input_dcd = f"{sysname}.dcd",
        #input_dcd = "combined.dcd",
        output_path = folder + '/data_first3.76mus',
        centered_dcd = "traj.dcd", 
        ref_name = ref_name, ref_chains = chaindict[ref_name],
        client_chain_list = client_chain_list, client_names = client_names,
        verbose = True)

    slab.center(end=7520,step=10, center_target='all')
    slab.calc_profiles()
    slab.calc_concentrations()
    slab.plot_density_profiles()

    
    slab_halftraj = cal.analysis.SlabAnalysis(
        name = name,
        input_path = folder,
        input_pdb = f"top.pdb",
        #input_dcd = f"{sysname}_first0.5mus.dcd",
        input_dcd = f"{sysname}.dcd",
        #input_dcd = "combined.dcd",
        output_path = folder + '/data_after0.5_first3.76mus',
        centered_dcd = "traj.dcd",
        ref_name = ref_name, ref_chains = chaindict[ref_name],
        client_chain_list = client_chain_list, client_names = client_names,
        verbose = True)

    slab_halftraj.calc_profiles(start = 100)
    slab_halftraj.calc_concentrations()
    slab_halftraj.plot_density_profiles()

