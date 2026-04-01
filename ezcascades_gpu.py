# initialise
import os
import sys
import json
import glob
import numpy as np
from scipy.spatial import cKDTree

import argparse, textwrap
from lib.descriptors.voxelised_descriptors import Voxel_Descriptors

from lib.eam_info import eam_info
from lib.lindhard import Lindhard, quickdamage
from lib.helperfuncs import sample_spherical, get_dump_frame, is_triclinic 

# load lammps module and style and variable types
from lammps import lammps


# template to replace MPI functionality for single threaded use
class MPI_to_serial():
    def bcast(self, *args, **kwargs):
        return args[0]
    def barrier(self):
        return 0


# Try running in parallel with MPI, otherwise fallback to serial
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    me = comm.Get_rank()
    nprocs = comm.Get_size()
    mode = 'MPI'

except ImportError:
    me = 0
    nprocs = 1
    mode = 'serial'


# Checking if all the GPU are visible - remove if there are no GPUs

# Print MPI rank info
print(f"[Rank {me}/{nprocs}] Mode: {mode}")

# Print visible GPU(s) for this rank
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "None")
print(f"[Rank {me}/{nprocs}] CUDA_VISIBLE_DEVICES: {cuda_visible}")

sys.stdout.flush()

def mpiprint(*arg):
    if me == 0:
        print(*arg)
        sys.stdout.flush()
    return 0



def announce(string):
    mpiprint ()
    mpiprint ("=================================================")
    mpiprint (string)
    mpiprint ("=================================================")
    mpiprint ()
    return 0 


kB = 8.617333262e-5

# better text formatting for argparse
class RawFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        return "\n".join([textwrap.fill(line, width) for line in textwrap.indent(textwrap.dedent(text), indent).splitlines()])

def main():
    program_descripton = f'''
        LAMMPS Simulation script for running overlapping cascades for alloys

        Max Boleininger, Aug 2024
        max.boleininger@ukaea.uk

        Licensed under the Creative Commons Zero v1.0 Universal
        https://creativecommons.org/publicdomain/zero/1.0/

        Distributed on an "AS IS" basis without warranties
        or conditions of any kind, either express or implied.

        USAGE:
        '''

    # -------------------
    # NUMBER OF GPUs
    # -------------------
   
    parser = argparse.ArgumentParser(description=program_descripton, formatter_class=RawFormatter)
    parser.add_argument("jsonfile", help="path to json input file.")
    parser.add_argument("-ngpu", "--numberofgpus", type=int, default=1, help="total number of GPUs to use for this simulation.")
    args = parser.parse_args()
    ngpu = args.numberofgpus

    # -------------------
    #  IMPORT PARAMETERS    
    # -------------------
  
    inputfile = args.jsonfile
    assert os.path.isfile(inputfile), "Error: input file %s not found." % inputfile  

    if (me == 0):
        with open(inputfile) as fp:
            all_input = json.loads(fp.read())
    else:
        all_input = None
    comm.barrier()

    # broadcast imported data to all cores
    all_input = comm.bcast(all_input, root=0)

    # -----------------------
    #  SET INPUT PARAMETERS 1   
    # -----------------------

    job_name = all_input['job_name']
    
    potdir  = all_input['potential_path']
    potname = all_input['potential']

    simdir = all_input["sim_dir"]
    scrdir = all_input["scratch_dir"]

    mpiprint ("Running in %s mode." % mode)
    mpiprint ("Job %s running on %s cores and %d GPUs per node.\n" % (job_name, nprocs, ngpu))
    
    mpiprint ("Parameter input file %s:\n" % inputfile)
    for key in all_input:
        mpiprint ("    %s: %s" % (key, all_input[key]))
    mpiprint()

    # If new run, clear previous relaxation files
    # else, look for restart file
    timestamp = None
    restartfile = None
    if me == 0:
        if all_input['simulation_clear'] == 1:
            for file in glob.glob("%s/%s/*.dump" % (scrdir, job_name)):
                os.remove(file)
            for file in glob.glob("%s/%s/*.dsc" % (scrdir, job_name)):
                os.remove(file)
            for file in glob.glob("%s/%s/*.restart" % (scrdir, job_name)):
                os.remove(file)
            for file in glob.glob("%s/log/%s.pka" % (simdir, job_name)):
                os.remove(file)
            for file in glob.glob("%s/log/%s.log" % (simdir, job_name)):
                os.remove(file)
        else:
            # fetch restart file
            restartpath = "%s/%s/%s.restart" % (scrdir, job_name, job_name)
            if os.path.exists(restartpath):
                announce ("Found restart file: %s" % restartpath)
            else:
                announce ("Casacade restart file %s not found. Starting new simulation." % restartpath)
                restartpath = None
            restartfile = restartpath
            
        if not os.path.exists("%s/%s" % (scrdir, job_name)):
            os.mkdir("%s/%s" % (scrdir, job_name))

    timestamp = comm.bcast(timestamp, root=0)
    restartfile = comm.bcast(restartfile, root=0)

    # -------------------
    #  INPUT POTENTIAL 
    # ------------------- 

    potfile = potdir + potname

    # Any EAM potential file will be scraped for lattice parameters etc
    potential = eam_info(potfile) # Read off

    mpiprint ('''Potential and elastic constants information:
    Elements, %s,
    mass: %s,
    lattice: %s,
    crystal: %s,
    cutoff: %s
    ''' % (
        potential.ele, potential.mass, 
        potential.lattice, potential.crystal, 
        potential.cutoff)
    )

    alattice = potential.lattice
    masses = all_input["masses"]
    znums = all_input["znums"]


    # -----------------------
    #  SET INPUT PARAMETERS 2   
    # -----------------------

    if potential.crystal == 'hcp':
        # if not supplied, initialise ideal c/a ratio for hcp lattice
        if "c_over_a" in all_input:
            c_over_a = all_input["c_over_a"]
        else:
            c_over_a = np.sqrt(8./3.)

        clattice = c_over_a * alattice 

        # LAMMPS lattice vectors for hcp lattice
        ix = np.r_[alattice, 0, 0]
        iy = np.r_[0, np.sqrt(3.)*alattice, 0]
        iz = np.r_[0, 0, clattice]

        # integer lattice repeats for simulation box size
        # here rescaled to give broadly similar dimensions for similar nx,ny,nz
        nx = np.round(all_input['nx'])
        ny = np.round(all_input['ny']/np.sqrt(3.))
        nz = np.round(all_input['nz']/np.sqrt(8./3.))

    else:
        # all INTEGER lattice vectors for LAMMPS lattice orientation
        ix = np.r_[all_input['ix']]
        iy = np.r_[all_input['iy']]
        iz = np.r_[all_input['iz']]

        nx = all_input['nx']
        ny = all_input['ny']
        nz = all_input['nz']


    # lattice vector norms
    sx = np.linalg.norm(ix)
    sy = np.linalg.norm(iy)
    sz = np.linalg.norm(iz)

    etol = float(all_input["etol"])
    etolstring = "%.5e" % etol

    # export every nth number of iterations
    export_nth = int(all_input["export_nth"])

    # maintain stresses during relaxation or during MD using barostat 
    if "boxstress" in all_input:
        boxstress = all_input["boxstress"]
    else:
        boxstress = {} 
    for sij in boxstress:
        boxstress[sij] = -boxstress[sij]*1e4 # convert GPa to bar

    # allow for the possibility of running from another starting point
    if "initial" in all_input:
        initial = all_input["initial"]
        initialtype = all_input["initialtype"]
    else:
        initial = None

    # temperature in kelvin
    temp = all_input["temperature"]
    if temp < 1e-9:
        athermal = True
    else:
        athermal = False

    if "langevin_damping_time" in all_input:
        tdamp = all_input["langevin_damping_time"] 
    else:
        tdamp = 5.0 # default value of 5 ps

    if "press_damping" in all_input:
        pdamp = all_input["barostat_damping_time"] 
    else:
        pdamp = 25.0 # default value of 25 ps

    # whether to run CG after each cascade propagation step
    runCG = all_input["runCG"]

    if runCG and not athermal:
        announce ('''WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
 
This is a finite temperature run where atomic coordinates are relaxed to a local
energy minimum with the conjugate gradient method after each cascade iteration:

\t('temp': %f with 'runCG': 1)!

This may lead to odd effects, as velocities are zeroed and thermal vibrations removed after every
cascade iteration. Consider running either an athermal simulation:

\t('temp': 0.0 with 'runCG': 1)

or a thermal simulation:

\t('temp'> 0.0 with 'runCG': 0)

WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING''' % temp)

    # ------------------------------
    #  COLLISION CASCADE PARAMETERS
    # ------------------------------

    # PKA events (eV) 
    PKAfile = all_input["PKAfile"] 

    # Import PKA events 
    if me == 0:
        mpiprint ("Importing PKA spectrum %s" % PKAfile)
        pkas = np.loadtxt(PKAfile, dtype=float)
        if pkas.shape == ():
            pkas = np.array([pkas])
        mpiprint ("Imported PKA spectrum of size %d events, min, mean, max energies: %6.4f %6.4f %6.4f" % (len(pkas), np.min(pkas), np.mean(pkas), np.max(pkas)))
        mpiprint ()
    else:
        pkas = None
    comm.barrier()
    pkas = comm.bcast(pkas, root=0)
    comm.barrier()

    # min threshold displacement energy (eV), cascade fragmentation energy (eV), and average displacement threshold energy (eV)
    pkamin = all_input["PKAmin"]
    pkamax = all_input["PKAmax"]
    edavg = all_input["edavg"]

    # minimum electronic stopping energy (eV)
    electronic_stopping_threshold = all_input["electronic_stopping_threshold"]

    # path to LAMMPS electronic stopping file 
    stoppingfile = all_input["stoppingfile"]

    # melting temperature, used to limit langevin fix to cold atoms and to estimate cascade heat spike size
    Tmelt = all_input['melting_temperature']

    # ensure the next cascade iteration does not proceed until the system has cooled below this temperature 
    tmax = all_input["maxtemperature"]

    # minimum propagation time in ps (useful for small cascades or to enforce a fixed dose-rate) 
    minruntime = all_input["minruntime"]

    # max dpa to propagate simulations for 
    if "maxdpa" in all_input:
        maxdpa = all_input["maxdpa"]
    else:
        maxdpa = 1.0

    # target dpa increment per cascade iteration (0.2 mpda results in an instantaneous temperature increase of 300K)
    if "incrementdpa" in all_input:
        incrementdpa = all_input["incrementdpa"]
    else:
        incrementdpa = 0.0002

    composition = {int(_type):all_input["composition"][_type] for _type in all_input["composition"].keys()}

    # we do not consider elements with zero contribution to the composition.
    # this filtering out is done to not load extraneous species, which would
    # otherwise also require the stopping file to contain info on this species
    present_species = [str(_key) for _key in composition if composition[_key] != 0.0]
    nelements = len(present_species)

    masses = {_key:masses[_key] for _key in present_species}
    znums = {_key:znums[_key] for _key in present_species}

    composition = {int(_key):composition[int(_key)] for _key in present_species}

    potential.ele = [potential.ele[int(i)-1] for i in present_species]


    # ----------------------------
    #  DETERMINE CELL DIMENSIONS    
    # ----------------------------

    # Check for right-handedness in basis
    if np.r_[ix].dot(np.cross(np.r_[iy],np.r_[iz])) < 0:
        mpiprint ("Left Handed Basis!\n\n y -> -y:\t",iy,"->",)
        for i in range(3):
            iy[i] *= -1
        mpiprint (iy,"\n\n")


    # Start LAMMPS instance
    lmp = lammps("gpu-cuda", cmdargs=["-sf", "gpu", "-pk", f"gpu {ngpu}"])

    lmp.command('# Lammps input file')
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')
    lmp.command('boundary p p p')
   
    # initialise lattice
    if potential.crystal == 'hcp':
        # hcp crystals are initialised such that box dimensions are orthogonal
        lmp.command('lattice custom 1.0 a1 %f %f %f a2 %f %f %f a3 %f %f %f basis 0.0 0.0 0.0 basis 0.5 0.5 0.0 basis 0.5 %f 0.5 basis 0.0 %f 0.5' % (
            ix[0], ix[1], ix[2],
            iy[0], iy[1], iy[2],
            iz[0], iz[1], iz[2],
            5./6., 1./3.))
    else:
        lmp.command('lattice %s %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % (
                    potential.crystal,
                    potential.lattice,
                    ix[0], ix[1], ix[2],
                    iy[0], iy[1], iy[2],
                    iz[0], iz[1], iz[2]))

    # cubic simulation cell region
    lmp.command('region r_simbox block 0 %d 0 %d 0 %d units lattice' % (nx, ny, nz))


    # read restart file and continue simulation from there, if available 
    if restartfile:
        announce("Restarting from last cascade file: %s" % restartfile)
        lmp.command('read_restart %s' % restartfile)
        
        # Set the time integration from the restart file to ensure continuity.
        if athermal:
            lmp.command('fix fnve all nve')
        else:
            # do not apply barostat if no stresses are set
            if boxstress == {}: 
                lmp.command('fix fnve all nve')

            else:
                nphstring = ''
                for _sij in boxstress:
                    nphstring += "%s %f %f %f " % (_sij, boxstress[_sij], boxstress[_sij], pdamp)
                lmp.command('fix fbaro all nph %s nreset 10' % nphstring) 

        # import log file and fetch last dose
        logdata = np.loadtxt('%s/log/%s.log' % (simdir, job_name))
        dpadose = logdata[-1,1]
        iteration = int(logdata[-1,0])
        lmp.command("print '# restart' append %s/log/%s.log" % (simdir, job_name))
        lmp.command("print '# restart' append %s/log/%s.pka" % (simdir, job_name))

    elif initial:
        # otherwise look for an initial file
        announce("Initialising structure from file: %s" % initial)

        # if neither initial nor restartfile have been given, initiate single crystal
        lmp.command('create_box %d r_simbox' % nelements) 

        if initialtype == "data":
            lmp.command('read_data %s' % initial)

        elif initialtype == "dump":
            initialframe = get_dump_frame(initial)

            tri, flag = is_triclinic (initial)
            if flag:
                announce (flag)
            if tri:
                lmp.command('run 0')
                lmp.command('change_box all triclinic')
            lmp.command('read_dump %s %d x y z purge yes add yes box yes replace no' % (initial, initialframe))
        
        lmp.command('reset_timestep 0')

    else:
        # if neither initial nor restartfile have been given, initiate single crystal
        lmp.command('create_box %d r_simbox' % nelements) 

        # initialise atoms as most commonly occuring species in the composition
        commontype = np.argmax(list(composition.values()))
        announce ("Initialising all atoms to the largest type: %d" % (1+commontype))
        lmp.command('create_atoms %d region r_simbox' % (1+commontype))
        natoms = lmp.extract_global("natoms", 0)

        if (me == 0):
            composition_array = np.random.choice(np.r_[:nelements], size=natoms, p=list(composition.values()))
        else:
            composition_array = None
        composition_array = comm.bcast (composition_array, root=0)

        # then set the remaining atomic species
        for _type in range(nelements):
            if _type == commontype:
                continue
            indices = tuple(1 + np.where(composition_array == _type)[0])
            announce ("Changing %d atoms to type: %d" % (len(indices), 1+_type))

            # work in batches, not setting too large groups a time
            maxgroupsize = 10000
            ngroups = int(len(indices)/maxgroupsize + 1)
            c = 0
            for _subindices in np.array_split(indices, ngroups):
                if len(_subindices) == 0:
                    continue
                mpiprint ("Batch %d out of %d..." % (c, ngroups))
                lmp.command('group gtype id' + " %d"*len(_subindices) % tuple(_subindices))
                lmp.command('set group gtype type %d' % (1+_type))
                lmp.command('group gtype delete')
                c += 1

    if not restartfile:
        dpadose = 0.0
        iteration = 0

    # load potential
    lmp.command('pair_style eam/%s' % potfile.split('.')[-1])
    lmp.command(('pair_coeff * * %s ' % potfile) + '%s '*nelements % tuple(potential.ele))

    # overwrite default masses
    for _i,_m in enumerate(masses.values()):
        lmp.command('mass %d %f' % (1+_i, _m)) 

    lmp.command('neighbor 3.0 bin')

    lmp.command('run 0')
    
    # time step in femtoseconds
    timestep = all_input["timestep"]
    lmp.command('timestep %f' % timestep)

    # thermo_style rate, same rate as updating Langevin damping group 
    nth = all_input["export_descriptors_nth"]

    # compute for evaluating per-atom kinetic energy, as used for dynamical Langevin damping group
    lmp.command('compute cke all ke/atom')
    lmp.command('compute cketot all reduce sum c_cke')

    # create a dynamical group of low kinetic energy atoms for applying Langevin thermostat
    damping_cutoff = 3*kB*Tmelt
    lmp.command('variable vekin atom "c_cke < %f"' % damping_cutoff)
    lmp.command('group gdamped dynamic all var vekin every %d' % nth)
    lmp.command('variable vndamped equal count(gdamped)')

    lmp.command('thermo %d' % nth)
    lmp.command('thermo_style custom step dt time temp press pe c_cketot v_vndamped pxx pyy pzz pxy pxz pyz lx ly lz')
    lmp.command("thermo_modify line one format line '%8d %5.3e %7.3f %6.3f %11.3e %15.8e %15.8e %8.0f %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %7.3f %7.3f %7.3f'")


    # if the simulation is not continuing from a restart file, relax structure and box dimensions
    if not restartfile:
        lmp.command('minimize %s 0 10000 10000' % (etolstring))

        rxstate = None
        if np.setdiff1d (["x","y","z","xy","xz","yz"], list(boxstress.keys())).size == 0 and np.sum(np.abs(list(boxstress.values()))) == 1e-9:
            # set as triclinic relaxation if all dimensions can relax
            lmp.command('fix ftri all box/relax tri 0.0 vmax 0.0001 nreset 100')
            rxstate = "tri"
        elif np.setdiff1d (["x","y","z"], list(boxstress.keys())).size == 0 and np.sum(np.abs(list(boxstress.values()))) < 1e-9:
            # set as orthorhombic relaxation if x,y,z dimensions can relax
            lmp.command('fix faniso all box/relax aniso 0.0 vmax 0.0001 nreset 100')
            rxstate = "aniso"
        else:
            # otherwise introduce multiple fixes
            for sij in boxstress:
                lmp.command('fix f%sfree all box/relax %s %f vmax 0.0001 nreset 100' % (sij, sij, boxstress[sij]))
            rxstate = "mixed"

        lmp.command('min_modify line quadratic')
        lmp.command('minimize %s 0 10000 10000' % (etolstring))

        # freeze box dimensions again
        if rxstate == "tri":
            lmp.command('unfix ftri') 
        elif rxstate == "aniso":
            lmp.command('unfix faniso')
        else:
            for sij in boxstress:
                lmp.command('unfix f%sfree' % sij)

        # wrap atoms back into the box
        lmp.command('reset_timestep 0')
        lmp.command('run 0')

        # print initial thermo quantities in log file
        lmp.command("variable vpe equal pe")
        lmp.command("variable vpxx equal pxx")
        lmp.command("variable vpyy equal pyy")
        lmp.command("variable vpzz equal pzz")
        lmp.command("variable vpxy equal pxy")
        lmp.command("variable vpxz equal pxz")
        lmp.command("variable vpyz equal pyz")
        lmp.command("variable vlx equal lx")
        lmp.command("variable vly equal ly")
        lmp.command("variable vlz equal lz")
        lmp.command("print '%d %f ${vpe} ${vpxx} ${vpyy} ${vpzz} ${vpxy} ${vpxz} ${vpyz} ${vlx} ${vly} ${vlz}' append %s/log/%s.log" % (iteration, 0.0, simdir, job_name))

        # for consistency, also write an essentially empty line into the pka file 
        lmp.command("print '%d 0.0' append %s/log/%s.pka" % (iteration, simdir, job_name))

    # initialise stopping model
    lmp.command('fix fstopping all electron/stopping %f %s' % (electronic_stopping_threshold, stoppingfile))

    lmp.command('thermo %d' % nth)
    lmp.command('thermo_style custom step dt time temp press pe c_cketot v_vndamped pxx pyy pzz pxy pxz pyz lx ly lz')
    lmp.command("thermo_modify line one format line '%8d %5.3e %7.3f %6.3f %11.3e %15.8e %15.8e %8.0f %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %7.3f %7.3f %7.3f'")

    # apply Langevin damping to dynamic group
    lmp.command('run 0')

    rndnumber = None
    if (me == 0):
        rndnumber = np.random.randint(1e8)
    rndnumber = comm.bcast(rndnumber, root=0)

    tdamp = all_input["langevin_damping_time"]        
    lmp.command('fix flangevin gdamped langevin %f %f %f %d' % (temp+1e-9, temp+1e-9, tdamp, rndnumber))

    # apply barostat if this is a thermal run with prescribed stress
    if not restartfile:
        if athermal:
            lmp.command('fix fnve all nve')
        else:
            # do not apply barostat if no stresses are set
            if boxstress == {}: 
                lmp.command('fix fnve all nve')

            else:
                nphstring = ''
                for _sij in boxstress:
                    nphstring += "%s %f %f %f " % (_sij, boxstress[_sij], boxstress[_sij], pdamp)
                lmp.command('fix fbaro all nph %s nreset 10' % nphstring) 
    
    # keep box centre of mass from drifting
    lmp.command("fix frecenter all recenter INIT INIT INIT")
          
    # initialise velocities and thermalise if this is a finite temperature run
    if not restartfile:
        if athermal:
            lmp.command('velocity all create 0.0 1 mom yes rot no')
        else:
            rndnumber = None
            if (me == 0):
                rndnumber = np.random.randint(1e8)
            rndnumber = comm.bcast(rndnumber, root=0)

            announce("Initialising random velocities with random number: %d" % rndnumber)
            lmp.command('velocity all create %f %d mom yes rot no' % (2*temp, rndnumber))

            # thermalisation here
            if "thermalisation_time" not in all_input:
                tthermal = 50.0 # default: 50 picoseconds
            else:
                tthermal = all_input["thermalisation_time"]

            announce ("Thermalising for %f picoseconds." % tthermal)
            lmp.command('run %d' % int(tthermal/timestep)) 
            
            # reset timestep and wrap atoms back into box
            lmp.command('reset_timestep 0')
            lmp.command('run 0')

            ###########################
            # ATOMIC DESCRIPTOR START #
            ########################### 

            # Initialise descriptors 
            voxeldescriptors = Voxel_Descriptors (all_input, lmp, nvox=all_input["voxelgrid"], mpi_comm=comm)

            # thermo commands
            lmp.command(f'thermo {voxeldescriptors.export_descriptors_nth}')
            lmp.command('thermo_style custom step time temp press pe ke pxx pyy pzz pxy pxz pyz lx ly lz')
            lmp.command("thermo_modify line one format line '%8d %7.3f %6.3f %11.3e %15.8e %15.8e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %7.3f %7.3f %7.3f'")

            lmp.command('run 0')
            lmp.command(f'run {voxeldescriptors.export_descriptors_nth} pre no post no') # propagate simulation (variables are not current after this)

            voxeldescriptors.are_all_atoms_accounted () # make variables current, check for internal consistency 
            voxeldescriptors.set_atoms_to_averaged ()   # replace momentary atomic coordinates with time-averaged coordinates

            if all_input["export_averaged_trajectories"]:
                lmp.command(f"write_dump all custom {scrdir}/{job_name}/{job_name}.ave.{iteration}.dump id type x y z")

            voxeldescriptors.evaluate () # evaluate voxel descriptors with time-averaged coordinates

            voxeldescriptors.export_computes (f"{scrdir}/{job_name}/{job_name}.{iteration}.dsc", "%d %f" % (iteration, 0.0), exporting_rank=0) # export descriptors to file
            voxeldescriptors.return_atoms_from_averaged ()  # restore atoms back to momentary coordinates
            voxeldescriptors.purge_all ()                   # clean-up in preparation for cascade run

            lmp.command('reset_timestep 0')
            lmp.command('run 0')

            #########################
            # ATOMIC DESCRIPTOR END #
            ######################### 


        # print out first dump
        if all_input['write_data']:
            lmp.command('write_dump all custom %s/%s/%s.%d.dump id type x y z' % (scrdir, job_name, job_name, iteration))




    # lindhard electronic stopping model for damage energy
    # build model averaged over the chance of occurrence of the specific projectile and target elements model 
    masslist = list(masses.values())
    zlist = list(znums.values())
    complist = list(composition.values())

    tdmodels = {}
    for i in range(nelements):
        for j in range(nelements):
            tdmodels[(i,j)] = Lindhard (masslist[i], masslist[j], zlist[i], zlist[j])

    # averaged model 
    def avg_tdmodel (Epka):
        tdenergy = 0.0
        for i in range(nelements):
            for j in range(nelements):
                tdenergy += complist[i]*complist[j]*tdmodels[(i,j)](Epka)
        tdenergy *= 1/sum(complist)**2
        return tdenergy

    tdmodel = np.vectorize (avg_tdmodel)

    # damage energy correction due to (typically) 10 eV cutoff in simulation
    dE = electronic_stopping_threshold - tdmodel (electronic_stopping_threshold)

    # estimate melting energy from 6*kB*Tmelt (overestimation as no latent heat of fusion given)
    emelt_guess = 6.*kB*Tmelt # in eV

    # define exclusion radius method 
    def exclusion_radius (PKAenergy):
        nmelt = PKAenergy/emelt_guess   # app. number of molten atoms

        if potential.crystal == 'bcc':
            vmelt = nmelt * .5*alattice**3  # app. melt volume in Ang^3
        if potential.crystal == 'fcc':
            vmelt = nmelt * .25*alattice**3  # app. melt volume in Ang^3

        rmelt = np.power(3./(4.*np.pi)*vmelt, 1./3.) # app. melt radius in Ang
        rbuffer = 5.0 # additional buffer in Ang
        exc_rad = rmelt + rbuffer # total exclusion radius in Ang
        return exc_rad
    exc_radius = np.vectorize (exclusion_radius)

    # run consecutive cascades 
    for cloop in range(1, int(1e6)):
   
        if dpadose >= maxdpa:
            announce ("Finished simulation. Current dose is %8.3f dpa, with target dose given by %8.3f dpa." % (dpadose, maxdpa))
            break 
 
        # extract cell dimensions 
        N = lmp.extract_global("natoms", 0)
        xlo, xhi = lmp.extract_global("boxxlo", 2), lmp.extract_global("boxxhi", 2)
        ylo, yhi = lmp.extract_global("boxylo", 2), lmp.extract_global("boxyhi", 2)
        zlo, zhi = lmp.extract_global("boxzlo", 2), lmp.extract_global("boxzhi", 2)
        xy, yz, xz = lmp.extract_global("xy", 2), lmp.extract_global("yz", 2), lmp.extract_global("xz", 2)

        # Relevant documentation: https://docs.lammps.org/Howto_triclinic.html 
        xlb = xlo + min(0.0,xy,xz,xy+xz)
        xhb = xhi + max(0.0,xy,xz,xy+xz)
        ylb = ylo + min(0.0,yz)
        yhb = yhi + max(0.0,yz)
        zlb, zhb = zlo, zhi

        lims_lower = np.r_[xlb, ylb, zlb]  # bounding box origin
        lims_upper = np.r_[xhb, yhb, zhb]  # bounding box corner
        lims_width = lims_upper-lims_lower # bounding box width

        # Basis matrix for converting scaled -> cart coords
        c1 = np.r_[xhi-xlo, 0., 0.]
        c2 = np.r_[xy, yhi-ylo, 0.]
        c3 = np.r_[xz, yz, zhi-zlo]
        cmat =  np.c_[[c1,c2,c3]].T
        cmati = np.linalg.inv(cmat)

        _x = np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(N, 3)
        _x = _x - np.array([xlo,ylo,zlo])

        # convert to fractional coordinates (einsum is faster than mat-mul) 
        _xfrac = np.einsum("ij,kj->ki", cmati, _x)

        # we need the IDs so that we know which atom type we are kicking 
        _ids = np.ctypeslib.as_array(lmp.gather_atoms("type", 0, 1))


        if (me == 0):
            # incremental applied dose (using damage energy)
            appdose = 0.0
            doselimit = incrementdpa 

            mpiprint ("Draw random cascade energies until the minimal dose increment is reached.")

            # upper limit: no more than 1.2 times dose limit
            nattempts = 0
            incrtol = 0.0 

            while (appdose >= (1.2+incrtol)*doselimit) or appdose == 0.0:
                appdose = 0.0
                cascade_pka = []

                # lower limit: higher than dose limit
                while (appdose <= max(0, (1.0-incrtol)*doselimit)):
                    epka = 0.
                    while (epka <= pkamin) or (epka > pkamax):
                        epka = np.random.choice(pkas)
                    cascade_pka += [epka]

                    # convert pka energies to damage energies
                    tdams = np.array([tdmodel(_pka)+dE for _pka in cascade_pka])

                    # compute frenkel pairs produced in NRT model and update dose increment
                    ndefects = np.array([quickdamage(_td, edavg) for _td in tdams])
                    appdose = np.sum(ndefects/N)

                nattempts += 1
                if nattempts == 1000:
                    print ()
                    print ("Could not draw damage energies within the range of [%3.2f*doseincrement, %3.2f*doseincrement] after %d attempts!" % (1.0-incrtol, 1.2+incrtol, nattempts))
                    print ("Proceeding to increase the acceptable dose range until damage energies can be drawn.")
                    print ()
                    
                if nattempts >= 1000:
                    incrtol += 0.1
                    
                if appdose > 1.5*incrementdpa:
                    print ("Error: could not accomodate recoil energise without exceeding 1.5 times the dose increment (c.f. %f vs %f dpa)!" % (appdose, incrementdpa))
                    print ("Your system might be too small to accomodate such large cascades. Consider increasing the system size.")
                    sys.stdout.flush()
                    return 0 

            print ("Final range: [%3.2f*doseincrement, %3.2f*doseincrement] after %d attempts." % (1.0-incrtol, 1.2+incrtol, nattempts))
            print ("Selecting recoils corresponding to a dose increment of: %f dpa" % appdose)
            
            # sort PKA energies in descending order to ensure we can fit in the largest cascades
            cascade_pka = np.flip(np.sort(cascade_pka))
            ncascades = len(cascade_pka)

            mpiprint ("Initialising %d cascades leading to a dose increment of %9.5f dpa with energies (eV):" % (ncascades, appdose))
            print ("damage energies:")
            print (cascade_pka)
            mpiprint ()

            cascade_pos = []
            mpiprint ("Drawing random non-overlapping cascade positions.")

            count = 0
            while (len(cascade_pos) < ncascades):

                # first, create a trial cascade inside the bounding box
                trisuccess = False
                for nattempts in range(int(1e6)):
                    _trial_pos = lims_width*np.random.rand(3)

                    # next, check if the point lies inside the triclinic box, otherwise repeat
                    _trial_pos
                    _frac = cmati@_trial_pos
                    if (_frac > 0.0).all() and (_frac < 1.0).all():
                        trisuccess = True
                        break
                
                if not trisuccess:
                    announce ("Error: could not place a random point inside triclinic cell after %d attempts." % nattempts)
                    return 1

                _ncasc = len(cascade_pos)
                if _ncasc == 0: 
                    # always accept the first cascade
                    cascade_pos += [_trial_pos]
                else:
                    # subsequent cascades need to be checked for overlap

                    # compute distance between trial cascade and all other cascades using minimum img convention
                    dr = cascade_pos - _trial_pos
                    df = np.array([cmati@dri for dri in dr])
                    df[df  > .5] -= 1.0
                    df[df <=-.5] += 1.0
                    dr = np.array([cmat@dfi for dfi in df]) # convert back to cartesian coordinates
                    dnorm = np.linalg.norm(dr, axis=1)

                    # get exclusion distances
                    excdist = exc_radius(cascade_pka[:_ncasc]) + exc_radius(cascade_pka[_ncasc])

                    # only accept cascade if all of the distances exceeed the exclusion distances
                    if (dnorm > excdist).all():
                        cascade_pos += [_trial_pos]

                if count >= 1e6:
                    mpiprint("Error: could not reach target dose after %d iterations (current dose: %8.5f)" % (count, appdose))
                    return 0
                count += 1

            cascade_pos = np.r_[cascade_pos]

            mpiprint ()
            mpiprint ("Selected cascade positions (Ang) and recoil energies (eV):")
            for _cp in range(ncascades):
                mpiprint ("(%8.3f, %8.3f, %8.3f)  %8.5f" % (cascade_pos[_cp][0], cascade_pos[_cp][1], cascade_pos[_cp][2], cascade_pka[_cp]))
            mpiprint ()

            # build KDTree (in fractional coords) for nearest neighbour search containing all atomic data
            xf_ktree = cKDTree(_xfrac, boxsize=[1,1,1])

            # find atoms nearest to cascade centres to apply kicks to
            kick_indices = [xf_ktree.query(cmati@_cpos, k=1)[1] for _cpos in cascade_pos]  # +1 for LAMMPS

            # kick velocities in Ang/ps (same as velocity in LAMMPS metal units)
            kickvelocities = np.array([np.sqrt(2.*cascade_pka[i]/(masses[str(_ids[ki])] * 1.03499e-4)) for i,ki in enumerate(kick_indices)])
            _sph = sample_spherical(ncascades)
            kick_velocities = [kickvelocities[_i]*_sph[_i] for _i in range(ncascades)]
        else:
            cascade_pka = None
            kick_indices = None
            kick_velocities = None 
            appdose = None
 

        comm.barrier()
        cascade_pka = comm.bcast(cascade_pka, root=0)
        kick_indices = comm.bcast(kick_indices, root=0)
        kick_velocities = comm.bcast(kick_velocities, root=0) 
        appdose = comm.bcast(appdose, root=0) 
        ncascades = len(cascade_pka)

        dpadose += appdose

        # apply random kicks
        for _c in range(ncascades):
            _ki = kick_indices[_c]
            mpiprint("Atom ID %d at (%8.3f, %8.3f, %8.3f) gets %12.6f eV recoil energy (%12.6f A/ps)." % (1+_ki,
                    _x[_ki][0]+xlo, _x[_ki][1]+ylo, _x[_ki][2]+zlo, cascade_pka[_c], np.linalg.norm(kick_velocities[_c])))
        
            lmp.command('group gkick id %d' % kick_indices[_c])
            lmp.command('velocity gkick set %f %f %f sum yes units box' % tuple(kick_velocities[_c]))
            lmp.command('group gkick delete') 
        mpiprint ()

        # initially strict adaptive time-step
        lmp.command('fix ftimestep all dt/reset 1 NULL 0.002 0.01 units box')
        lmp.command('run 500 post no')

        # less strict later on 
        lmp.command('unfix ftimestep')
        lmp.command('fix ftimestep all dt/reset 10 NULL 0.002 0.1 units box')
            
        maxloops = int(1e6)
        nsteps = 250
        announce("Running loops of %d steps while monitoring temperature." % nsteps)
           
        # simulation time value is not reset, so need to keep track of the difference 
        initial_time = float(lmp.get_thermo('time'))

        for k in range(maxloops):
            lmp.command('run %d post no' % nsteps) 
            current_temperature = float(lmp.get_thermo('temp')) 
            current_time = float(lmp.get_thermo('time'))

            if current_temperature <= tmax:
                if (current_time-initial_time) > minruntime:
                    mpiprint("Reached maximal temperature and minimum run time, propagation complete.")
                    break
                else:
                    mpiprint("Reached maximal temperature but not minimum run time, propagating more.")


        # if required, finish the iteration with CG minimisation
        if runCG:
            # zero out velocities
            lmp.command('velocity all create 0.0 1 mom yes rot no')
 
            # relax atomic coordinates only
            lmp.command('minimize %s 0 10000 10000' % (etolstring))

            rxstate = None
            if np.setdiff1d (["x","y","z","xy","xz","yz"], list(boxstress.keys())).size == 0 and np.sum(np.abs(list(boxstress.values()))) == 1e-9:
                # set as triclinic relaxation if all dimensions can relax
                lmp.command('fix ftri all box/relax tri 0.0 vmax 0.0001 nreset 100')
                rxstate = "tri"
            elif np.setdiff1d (["x","y","z"], list(boxstress.keys())).size == 0 and np.sum(np.abs(list(boxstress.values()))) < 1e-9:
                # set as orthorhombic relaxation if x,y,z dimensions can relax
                lmp.command('fix faniso all box/relax aniso 0.0 vmax 0.0001 nreset 100')
                rxstate = "aniso"
            else:
                # otherwise introduce multiple fixes
                for sij in boxstress:
                    lmp.command('fix f%sfree all box/relax %s %f vmax 0.0001 nreset 100' % (sij, sij, boxstress[sij]))
                rxstate = "mixed"

                lmp.command('min_modify line quadratic')
                lmp.command('minimize %s 0 10000 10000' % (etolstring))

                # freeze box dimensions again
                if rxstate == "tri":
                    lmp.command('unfix ftri') 
                elif rxstate == "aniso":
                    lmp.command('unfix faniso')
                else:
                    for sij in boxstress:
                        lmp.command('unfix f%sfree' % sij)
        elif not athermal:
            # otherwise just rescale velocities to avoid temperature drift
            lmp.command('velocity all scale %f' % temp)


        # wrap atoms back into the box
        lmp.command('unfix ftimestep')
        lmp.command('reset_timestep 0')
        lmp.command('run 0')

        ###########################
        # ATOMIC DESCRIPTOR START #
        ########################### 

        # Initialise descriptors 
        voxeldescriptors = Voxel_Descriptors (all_input, lmp, nvox=all_input["voxelgrid"], mpi_comm=comm)

        # thermo commands
        lmp.command(f'thermo {voxeldescriptors.export_descriptors_nth}')
        lmp.command('thermo_style custom step time temp press pe ke pxx pyy pzz pxy pxz pyz lx ly lz')
        lmp.command("thermo_modify line one format line '%8d %7.3f %6.3f %11.3e %15.8e %15.8e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %7.3f %7.3f %7.3f'")

        lmp.command('run 0')
        lmp.command(f'run {voxeldescriptors.export_descriptors_nth} pre no post no') # propagate simulation (variables are not current after this)

        voxeldescriptors.are_all_atoms_accounted () # make variables current, check for internal consistency 
        voxeldescriptors.set_atoms_to_averaged ()   # replace momentary atomic coordinates with time-averaged coordinates

        if ((iteration + cloop) % voxeldescriptors.export_trajectories_nth) == 0:
            if all_input["export_averaged_trajectories"]:
                lmp.command(f"write_dump all custom {scrdir}/{job_name}/{job_name}.ave.{iteration+cloop}.dump id type x y z")

        voxeldescriptors.evaluate () # evaluate voxel descriptors with time-averaged coordinates

        voxeldescriptors.export_computes (f"{scrdir}/{job_name}/{job_name}.{iteration+cloop}.dsc", "%d %f" % (iteration+cloop, dpadose), exporting_rank=0) # export descriptors to file
        voxeldescriptors.return_atoms_from_averaged ()  # restore atoms back to momentary coordinates
        voxeldescriptors.purge_all ()                   # clean-up in preparation for cascade run

        lmp.command('reset_timestep 0')
        lmp.command('run 0')

        #########################
        # ATOMIC DESCRIPTOR END #
        ######################### 

        # print thermo quantities in log file
        lmp.command("variable vpe equal pe")
        lmp.command("variable vpxx equal pxx")
        lmp.command("variable vpyy equal pyy")
        lmp.command("variable vpzz equal pzz")
        lmp.command("variable vpxy equal pxy")
        lmp.command("variable vpxz equal pxz")
        lmp.command("variable vpyz equal pyz")
        lmp.command("variable vlx equal lx")
        lmp.command("variable vly equal ly")
        lmp.command("variable vlz equal lz")
        lmp.command("print '%d %f ${vpe} ${vpxx} ${vpyy} ${vpzz} ${vpxy} ${vpxz} ${vpyz} ${vlx} ${vly} ${vlz}' append %s/log/%s.log" % (iteration+cloop, dpadose, simdir, job_name))

        # append to recoil energy log file
        pka_string = "print '%d " % (iteration+cloop) + "%10.6f "*len(cascade_pka) % tuple(cascade_pka) +  "' append %s/log/%s.pka" % (simdir, job_name)
        lmp.command(pka_string)

        # write restart file always - this is in data format for also reading velocities
        dfile = "%s/%s/%s.restart" % (scrdir, job_name, job_name)
        announce("Writing restart file: %s" % dfile)
        lmp.command('write_restart %s' % dfile) 

        # write dump file every 'export_nth' steps
        if ((iteration + cloop) % export_nth) == 0:
            dfile = "%s/%s/%s.%d.dump" % (scrdir, job_name, job_name, iteration+cloop)
            announce("Writing dump file %s." % dfile)
            lmp.command('write_dump all custom %s id type x y z' % dfile)     

        comm.barrier()

    lmp.close()

    return 0


if __name__ == "__main__":
    main()

    if mode == 'MPI':
        MPI.Finalize()





