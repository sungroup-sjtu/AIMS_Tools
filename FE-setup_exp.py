import sys
import os
import shutil

sys.path.append('/share/workspace/eva/HTP/AIMS_Tools') #directory path
#print('hello')

from mstools.simulation.gmx import HydraFe
#print('hello')
from mstools.jobmanager import Slurm

slurm=Slurm(*('cpu', 16, 0, 16), env_cmd='module purge; module load gromacs/2016.6')
hydra_fe = HydraFe(packmol_bin='/share/apps/tools/packmol',
          dff_root='/home/gongzheng/apps/DFF/Developing',
          dff_db='TEAMFF',
          dff_table='TEAM_LS',
          gmx_bin='gmx_serial',
          gmx_mdrun='gmx_gpu mdrun',
          jobmanager=slurm)

smiles = ['C', 'O']
Nmol = [1, 600]

#print(smiles, type(smiles))
#print(Nmol, type(Nmol))

T = 298
P = 1
jobname='%s-%i-%i' %(smiles, T, P)

hydra_fe.set_system(smiles, n_mol_list=Nmol)
print('system set')
hydra_fe.build()
print('box built')

with open('topol.itp', 'r+') as f_in:
    lines = f_in.readlines()

with open('topol.itp', 'w') as f_out:
    for line in lines:
        #print(line, type(line))
        if 'o_2w       o_2w' in line:
            continue
        else:
            f_out.write(line)

hydra_fe.prepare(drde=False, T=T, P=P, nst_eq=int(2E5), nst_run=int(3E5), nst_xtc=500, jobname=jobname)
print('simu prepared')
hydra_fe.clean()
# print('clean up')

#Calculation set up

print('setting up FE simalation')
lambda_vec = [0.00, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80,
                       1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
src = os.getcwd()

files = ['topol.top', 'topol.itp', 'conf.gro', '_job_slurm.sh', 'grompp-alq-em.mdp', 'grompp-embox.mdp',
         'grompp-npt-eq.mdp', 'grompp-nvt-eq.mdp', 'grompp-npt-prod.mdp']
FE_mdp_temp = ['grompp-alq-em.mdp', 'grompp-npt-eq.mdp', 'grompp-nvt-eq.mdp', 'grompp-npt-prod.mdp']
FE_mdp_out = ['alq_em.mdp', 'npt_eq.mdp', 'nvt_eq.mdp', 'npt_prod.mdp']

lambda_points = {}
names = []
temp_paths = []
for i in range(len(lambda_vec)):
    name ='lambda-%i' % i
    os.mkdir(name)
    names.append(name)
    for nm in names:
        dest = os.path.abspath(nm)
    lambda_points[i] = name, dest

for point in lambda_points:
    lambda_dir, lambda_dest = lambda_points[point]
    for file in files:
        shutil.copy(file, lambda_dest)
    os.chdir(lambda_dest)
    for mdp_temp in FE_mdp_temp:
        with open(mdp_temp, 'r') as mdp_in:
            contents = mdp_in.read()
            contents = contents.replace('%lambda%', str(point))
        temp_paths.append(os.path.join(lambda_dest, mdp_temp))
    for mdp in FE_mdp_out:
        with open(mdp, 'w') as f_out:
            f_out.write(contents)

for path in temp_paths:
    os.remove(path)

for pnt in lambda_points:
    dir, dest = lambda_points[pnt]
    os.chdir(dest)
    os.rename('alq_em.mdp', 'grompp-alq-em.mdp')
    os.rename('npt_eq.mdp', 'grompp-npt-eq.mdp')
    os.rename('nvt_eq.mdp', 'grompp-nvt-eq.mdp')
    os.rename('npt_prod.mdp', 'grompp-npt-prod.mdp')
    #os.system('sbatch _job_slurm.sh') #submission
