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
    parser.add_argument('--start',type=int)
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
    
    slab_full = cal.analysis.SlabAnalysis(
        name = name,
        input_path = folder,
        input_pdb = f"top.pdb",
        #input_dcd = f"{sysname}_first0.5mus.dcd",
        input_dcd = f"{sysname}.dcd",
        #input_dcd = "combined.dcd",
        output_path = folder + '/data_first0.5mus',
        centered_dcd = f'traj_first0.5mus.dcd', 
        ref_name = ref_name, ref_chains = chaindict[ref_name],
        client_chain_list = client_chain_list, client_names = client_names,
        verbose = True)

    slab_full.center(start=0,end = 5000,step=10)
    slab_full.calc_profiles()
    slab_full.calc_concentrations()
    slab_full.plot_density_profiles()

    
    # slab_after05 = cal.analysis.SlabAnalysis(
    #     name = name,
    #     input_path = folder,
    #     input_pdb = f"top.pdb",
    #     #input_dcd = f"{sysname}_first0.5mus.dcd",
    #     input_dcd = f"{sysname}.dcd",
    #     #input_dcd = "combined.dcd",
    #     output_path = folder + '/data_after0.5',
    #     centered_dcd = "traj_full.dcd",
    #     ref_name = ref_name, ref_chains = chaindict[ref_name],
    #     client_chain_list = client_chain_list, client_names = client_names,
    #     verbose = True)

    # slab_after05.calc_profiles(start = 100)
    # slab_after05.calc_concentrations()
    # slab_after05.plot_density_profiles()


    # slab_last2mus = cal.analysis.SlabAnalysis(
    # name = name,
    # input_path = folder,
    # input_pdb = f"top.pdb",
    # #input_dcd = f"{sysname}_first0.5mus.dcd",
    # input_dcd = f"{sysname}.dcd",
    # #input_dcd = "combined.dcd",
    # output_path = folder + '/data_last2mus_full',
    # centered_dcd = "traj_full.dcd",
    # ref_name = ref_name, ref_chains = chaindict[ref_name],
    # client_chain_list = client_chain_list, client_names = client_names,
    # verbose = True)

    # slab_last2mus.calc_profiles(start = -int(4000/step))
    # slab_last2mus.calc_concentrations()
    # slab_last2mus.plot_density_profiles()
