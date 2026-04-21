import os
from calvados.cfg import Config, Job, Components
import shutil
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from Bio import SeqIO

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=False,type=str,default = "NUP98")
parser.add_argument('--path',nargs = '?', required = False, type = str,default = "hremd-test_NUP98")
parser.add_argument('rtladder',required = False, type = str,default = "0,0.75,0.79,0.81")
parser.add_argument('--temp',required = False, type = float,default = 270.15)
parser.add_argument('--platform', required = False, default="CUDA")
args = parser.parse_args()

t = float(args.temp)
rts = [float(i) for i in args.rtladder.split(",")]

cwd = os.getcwd()
N_save = int(5e4)
N_tot = int(100000) #uses sim_time in mus 
N_frames = int(N_tot/N_save)
N_exchange = 10000
N_batches = N_tot//N_exchange

for r in rts:
  sysname = f'{args.name:s}_rt{r}sig_{t:.2f}K'
  
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
    restart = 'checkpoint',
    frestart = 'restart.chk',
    verbose = True,
    slab_eq = True,
    steps_eq = int(5e3), 
    logfreq = N_exchange,
    sc_mode = "rt2",
    rt = r,
    softcore = True
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
