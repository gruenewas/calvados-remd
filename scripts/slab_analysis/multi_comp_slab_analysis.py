import calvados as cal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import MDAnalysis as mda

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--name',type=str)
    parser.add_argument('--sysname',type=str)
    #parser.add_argument('-sim_time',type=str,default="9")
    parser.add_argument('--path',type=str)
   # parser.add_argument('--replica',type=str,default="1")
    parser.add_argument('--ref_name',type=str)
    parser.add_argument('--start',type=int)
    parser.add_argument('--step',type=int,required=False,default=10)
    args = parser.parse_args()
   #start = {"260.15":int(20e3),"280.15":int(22e3),"290.15":int(20e3),"310.15":int(21e3)}
    name = args.name
    sysname = args.sysname
    folder = f"{args.path}/{sysname}"
    ref_name = args.ref_name
    print(sysname)
    #chaindict = {"Lin-13":(0,39),"Hpl-2":(40,79)}
#    u = mda.Universe(f"top_reindexed.pdb",f"{sysname}.dcd")
    #n_frames = len(u.trajectory)
    #start = int(n_frames - args.keep) 
    #chaindict = fix_trajectory(path=folder,simlog_path=f"{folder}/placement.txt",mixed_pdb="equilibration_final.pdb",mixed_dcd=f"{sysname}.dcd",overwrite=True,start=0,step=10)
    #chaindict = fix_trajectory(path=folder,simlog_path=f"{folder}/placement.txt",mixed_pdb="equilibration_final.pdb",mixed_dcd=f"{sysname}.dcd",overwrite=True,start=args.keep,step=10,out_dcd="traj_reindexed_last4mus.dcd")
    #plot_e2e_corr_function(folder=folder,sysname=name,temp=args.T,comp_dict=chaindict,out_path=f"{folder}/data",factor=10)
    chaindict = {"Lin-13":(0,39),"Hpl-2":(40,79)}
    client_names = [i for i in chaindict.keys() if i != ref_name]
    client_chain_list = [chaindict[i] for i in client_names]
    slab = cal.analysis.SlabAnalysis(
        name = name,
        input_path = folder,
        input_pdb = "top.pdb",
        input_dcd = f"{sysname}.dcd",
        output_path = folder + '/data',
        centered_dcd = f"traj_full.dcd",
        ref_name = ref_name, ref_chains = chaindict[ref_name],
        client_chain_list = client_chain_list, client_names = client_names,
        verbose = True)

    slab.center(start=args.start, step=args.step, center_target='ref')
    slab.calc_profiles()
    slab.calc_concentrations()
    slab.plot_density_profiles()
    #print("Calculating chain MSDs ")
    #compute_chain_msd(path = folder, top_file = "top_reindexed.pdb", traj_file = "traj_full.dcd", res_path = "input/residues_CALVADOS3.csv",out_path = f"folder/data",components=chaindict,filter_inside=True,results_path=f"{folder}/data/hpl2+lin13_ps_results.csv")
