import sys
import os
import shutil
sys.path.append('/lustre/home/acct-nishsun/nishsun-echarvati/HTP/AIMS_Tools/')
from mstools.simulation.gmx import HydraFe
from mstools.jobmanager import Slurm



def runFE(mol_list="mols.in", hydra=True, octanol=True):

    slurm=Slurm(*('small', 12, 0, 12), env_cmd='module purge\n\
    module load gromacs/2019.4-gcc-9.2.0-openmpi\n\
    ulimit -s unlimited\n\
    ulimit -l unlimited\n\
    export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so\n\
    export I_MPI_FABRICS=shm:ofi\n\
    ')

    hydra_fe = HydraFe(packmol_bin='/lustre/home/acct-nishsun/nishsun-echarvati/apps/packmol/packmol',
            dff_root='/lustre/home/acct-nishsun/nishsun-echarvati/apps/dff/',
            dff_db='TEAMFF',
            dff_table='TEAM_LS',
            gmx_bin='gmx_mpi',
            gmx_mdrun='--mpi=pmi2 gmx_mpi mdrun',
            jobmanager=slurm)

    workspace = os.getcwd()
    mols = []
    fin=open(mol_list,'r')
    for line in fin:
        mols.append(line.strip().split('\n'))
    fin.close()
    print(mols)

    if hydra==True:
        solv = "O"
        for smiles in mols:
            smiles.append(solv)
            Nmol = [1, 1000]
            print(smiles, smiles[0], type(smiles[0]), type(smiles))
            print(Nmol, type(Nmol))
            mol = smiles[0]
            moldir = os.path.join(workspace, mol)
            os.mkdir(moldir)
            os.chdir(moldir)
            T = 298
            P = 1
            TANNEAL= None
            lambdas = 20

            hydra_fe.set_system(smiles, n_mol_list=Nmol)
            hydra_fe.build()
            print('box built')

            hydra_fe.FF_cleanup(itp_in='topol.itp')
            print('o_2w - o_2w nobonding interactions excluded')

            hydra_fe.prepare(drde=False, T=T, P=P, TANNEAL=TANNEAL, nst_eq=int(2E5), nst_run=int(3E5), nst_xtc=500,
                             jobname='1_%s-O-300' %(mol))
            print('simu prepared')
            # hydra_fe.clean()

            hydra_fe.filemanager(TANNEAL=TANNEAL, lambdas=lambdas)


            print(os.getcwd())




runFE(mol_list="mols.in", hydra=True, octanol=False)

