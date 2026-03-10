import os
from calvados.cfg import Config, Job, Components
import shutil
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from Bio import SeqIO

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=False,type=str,default = "NUP98")
parser.add_argument('--path',nargs = '?', required = False, type = str,default = "tremd-test_NUP98")
parser.add_argument('--lowT', required = False, type = float,default = 290.15)
parser.add_argument('--highT', required = False, type = float,default = 293.15)
parser.add_argument('--nTemps',required = False, type = int,default = 4)
parser.add_argument('--platform', required = False, default="CPU")
args = parser.parse_args()


temprange = np.round(np.linspace(args.lowT,args.highT,args.nTemps),2)
n_threads = (2 if len(temprange) <= 4 else 1)

cwd = os.getcwd()
N_save = int(5e4)
N_tot = int(100000) #uses sim_time in mus 
N_frames = int(N_tot/N_save)
N_exchange = 10000
N_batches = N_tot//N_exchange

for t in temprange:
  sysname = f'{args.name:s}_{t:.2f}K'
  
  config = Config(
    # GENERAL
    sysname = sysname, # name of simulation system
    box = [30, 30, 300.], # nm
    temp = float(t),
    ionic = 0.15, # molar
    pH = 7,
    topol = 'slab',
    slab_width = 40,
    friction = 0.01,
    report_potential_energy = True,
    custom_restraints = "hpl" in sysname.lower(),
    fcustom_restraints = "custom_restraints.txt",
    

    # RUNTIME SETTINGS
    gpu_id = 0,
    wfreq = N_exchange, # N_save dcd writing frequency, 1 = 10fs
    steps = N_exchange,
    total_steps = N_tot,#N_frames*N_save number of simulation steps
    batches = N_batches, 
    runtime = 0, # overwrites 'steps' keyword if > 0
    platform = args.platform, # 'CUDA'
    threads = n_threads,
    restart = 'checkpoint',
    frestart = 'restart.chk',
    verbose = True,
    slab_eq = True,
    steps_eq = int(5e3), 
    logfreq = N_exchange
    )

  # PATH
  path = f'{cwd}/{args.path:s}/{sysname}'
  output_path = f'{path}/data'
  Path(path).mkdir(parents=True, exist_ok=True)
  Path(output_path).mkdir(parents=True, exist_ok=True)

  analyses = f"""
  """
  config.write(path,name='config.yaml',analyses=analyses)

  components = Components(
    # Defaults
    molecule_type = 'protein',
    nmol = 1, # number of molecules
    charge_termini = 'both', # charge N or C or both

    # INPUT
    ffasta = f'{cwd}/input/mix.fasta', # input fasta file
    fdomains = f'{cwd}/input/domains.yaml',
    )

  components.add(name= "NUP98_WT", nmol=1, fresidues = f'{cwd}/input/residues_CALVADOS2.csv', restraint=False)
  #components.add(name="hPol2_fl",nmol=10, fresidues = f'{cwd}/input/residues_CALVADOS3.csv', restraint=True, pdb_folder = f'{cwd}/input')
    #src = Path(cwd) / "input" / "custom_restraints.txt"
    #dst = Path(path) / src.name 
    #shutil.copy2(src, dst)


  components.write(path,name='components.yaml')
