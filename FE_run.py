import sys
import os
import shutil

sys.path.append('/share/workspace/eva/HTP/AIMS_Tools')
from mstools.simulation.gmx import HydraFe
from mstools.jobmanager import Slurm

slurm=Slurm(*('gtx', 8, 1, 8), env_cmd='module load gromacs/2018.1')
hydra_fe = HydraFe(packmol_bin='/share/apps/tools/packmol',
          dff_root='/home/gongzheng/apps/DFF/Developing',
          dff_db='TEAMFF',
          dff_table='TEAM_LS',
          gmx_bin='gmx_serial',
          gmx_mdrun='gmx_gpu mdrun',
          jobmanager=slurm)

smiles = ['CCC=C', 'O']
Nmol = [1, 1000]

print(smiles, type(smiles))
print(Nmol, type(Nmol))

T = 298
P = 1
TANNEAL= None
lambdas = 20

hydra_fe.set_system(smiles, n_mol_list=Nmol)
print('system set')
hydra_fe.build()
print('box built')

hydra_fe.FF_cleanup(itp_in='topol.itp')
print('o_2w - o_2w nobonding interactions excluded')

hydra_fe.prepare(drde=False, T=T, P=P, TANNEAL=TANNEAL, nst_eq=int(2E5), nst_run=int(3E5), nst_xtc=500, jobname='1_C4H8-O-300-%lambda%')
print('simu prepared')
#hydra_fe.clean()

hydra_fe.filemanager(TANNEAL=TANNEAL, lambdas=lambdas)
