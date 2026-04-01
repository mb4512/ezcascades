import numpy as np
import sys

from lib.affine import getcell, AffineTransform

from lammps import LMP_VAR_EQUAL, LMP_VAR_ATOM, LMP_VAR_VECTOR, LMP_VAR_STRING
from lammps import LAMMPS_INT, LAMMPS_INT_2D, LAMMPS_DOUBLE, LAMMPS_DOUBLE_2D, LAMMPS_INT64, LAMMPS_INT64_2D, LAMMPS_STRING

from lammps import LMP_STYLE_GLOBAL, LMP_STYLE_ATOM, LMP_STYLE_LOCAL
from lammps import LMP_TYPE_SCALAR, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY, LMP_SIZE_VECTOR, LMP_SIZE_ROWS, LMP_SIZE_COLS 

class _Voxel:
    def __init__(self, lmp, nvox=[1,1,1], mpi_comm=None):

        self.comm = mpi_comm
        if mpi_comm:
            self.me = self.comm.Get_rank()
        else:
            self.me = 0
        
        self.tridims = ["xlo", "xhi", "ylo", "yhi", "zlo", "zhi", "xy", "xz", "yz"]

        # make sure we always start at timestep 0
        lmp.command("reset_timestep 0")

        # define variabels and regions associated with voxels
        nvoxels = np.prod(nvox)
        
        voxelbounds = {}
        for ni in range(nvox[0]):
            for nj in range(nvox[1]):
                for nk in range(nvox[2]):
                    _vid=f"v_{ni}_{nj}_{nk}" # variable identifier 
                    _rid=f"r_{ni}_{nj}_{nk}" # region identifier
                    vb={}
                    vb["xlo"], vb["xhi"] = f"xlo+(xhi-xlo)*{ni/nvox[0]}", f"xlo+(xhi-xlo)*{(ni+1)/nvox[0]}"
                    vb["ylo"], vb["yhi"] = f"ylo+(yhi-ylo)*{nj/nvox[1]}", f"ylo+(yhi-ylo)*{(nj+1)/nvox[1]}"
                    vb["zlo"], vb["zhi"] = f"zlo+(zhi-zlo)*{nk/nvox[2]}", f"zlo+(zhi-zlo)*{(nk+1)/nvox[2]}"
                    vb["xy"], vb["xz"], vb["yz"] = f"xy/{nvox[0]}", f"xz/{nvox[1]}", f"yz/{nvox[2]}"

                    # for each voxel, define a variable storing its triclinic dimensions 
                    for _s in self.tridims: 
                        lmp.command(f'variable {_vid}_{_s} equal {vb[_s]}')

                    # define dynamically resizing regions based on the voxel variables
                    lmp.command(f"region {_rid} prism v_{_vid}_xlo v_{_vid}_xhi v_{_vid}_ylo v_{_vid}_yhi v_{_vid}_zlo v_{_vid}_zhi v_{_vid}_xy v_{_vid}_xz v_{_vid}_yz units box")

                    # store voxel boundaries and region ID in dictionary for later retrieval
                    voxelbounds[_vid] = vb

        lmp.command('run 0')

        # the v_globalcount variable tallies the total number of atoms contained in voxels (=should be equal to natoms)
        # test for internal consistency: the voxels should exactly contain all atoms of the system
        sumstring="variable v_globalcount equal "
        for _vid in voxelbounds:
            _rid = "r_%s" % _vid.lstrip("v_")
            lmp.command(f"variable v_count_{_rid} equal count(all,{_rid})")
            sumstring += f"v_v_count_{_rid}+"
        sumstring = sumstring.rstrip("+")
        lmp.command(sumstring)

        # store important values as properties of this class
        self.lmp = lmp
        self.nvoxels = nvoxels
        self.voxelbounds = voxelbounds


    def are_all_atoms_accounted (self):
        '''Check for internal consistency: are all atoms included in the voxels?'''
        lmp = self.lmp

        lmp.command("run 0") # always use up to date variables
        voxeltally = int(lmp.extract_variable('v_globalcount'))
        natoms = lmp.extract_global('natoms', dtype=LAMMPS_INT)

        if voxeltally == natoms:
            return True
        else:
            mpiprint ("ERROR: Not all atoms accounted for in voxels.")
            return False


    def set_computes (self, compute_string, compute_format):
        self.compute_string = compute_string
        self.compute_format = compute_format 

    def _check_compute (self):
        assert hasattr(self, "compute_string"), "ERROR: define compute with set_computes first."
        assert hasattr(self, "compute_format"), "ERROR: define compute with set_computes first."
        return 0

    def evaluate (self): 
        # set and evaluate voxelised computes
        lmp = self.lmp
        self._check_compute()

        for computeID in self.compute_string:
            computeStr = self.compute_string[computeID]
            lmp.command(f"compute {computeID} all {computeStr}") 

        # sum computes over voxel regions
        for computeID in self.compute_string:
            for _vid in self.voxelbounds:
                _rid = "r_%s" % _vid.lstrip("v_")
                _cformat = self.compute_format[computeID]
                lmp.command(f"compute ave_{computeID}_{_rid} all reduce/region {_rid} sum c_{computeID}{_cformat}")

        # evaluate computes 
        lmp.command("run 0")


    def purge_voxel_computes (self):
        # unset computes (use for continuous runs)
        lmp = self.lmp
        self._check_compute()

        for computeID in self.compute_string: 
            for _vid in self.voxelbounds:
                _rid = "r_%s" % _vid.lstrip("v_")
                lmp.command(f"uncompute ave_{computeID}_{_rid}")
            lmp.command(f"uncompute {computeID}")

        lmp.command("run 0")

    def purge_all (self):
        # unset all computes and fixes (use this for single-shot runs)
        lmp = self.lmp

        self.purge_voxel_computes()

        lmp.command("unfix ave")
        lmp.command("uncompute cunw")
        lmp.command("uncompute cimg")

        for _vid in self.voxelbounds:
            _rid = "r_%s" % _vid.lstrip("v_")
            lmp.command(f"variable {_vid} delete")
            lmp.command(f"region {_rid} delete")

        lmp.command("run 0")

    def export_computes (self, filepath, frameID, exporting_rank=0):
        '''Extract computes to file.
        
        Format:
        =======
        INT, 1x1: frame, or some other unique time identifier
        INT, 1x2: number of voxels (=NVOX), number of compute coefficients (=NC)
        FLOAT (NVOX,9): xlo xhi ylo yhi zlo zhi xy yz xz natoms, dimensions (Angstrom) and number of atoms per voxel
        STRING: compute instructions
        FLOAT (NVOX,NC): compute coefficients, one row per voxel
        '''
        lmp = self.lmp
        self._check_compute()

        if (self.me == exporting_rank):
            print ("**************************************************")
            print (f"Exporting computes to file {filepath}")
            print ("**************************************************")
        sys.stdout.flush ()

        voxelbounds_export = []
        for _vid in self.voxelbounds:
            _row = []
            _rid = "r_%s" % _vid.lstrip("v_")
            for _s in self.tridims:
                _row += [lmp.extract_variable(f"{_vid}_{_s}", vartype=LMP_VAR_EQUAL)]
            _row += [lmp.extract_variable(f"v_count_{_rid}", vartype=LMP_VAR_EQUAL)]
            voxelbounds_export += [_row]


        compute_header = ""
        compute_export = np.empty((self.nvoxels,0))
        for computeID in self.compute_string:
            _compute_export = []
            for _nvi,_vid in enumerate(self.voxelbounds):
                _rid = "r_%s" % _vid.lstrip("v_")
                _vb = [self.voxelbounds[_vid][_k] for _k in self.tridims]

                if "[" in self.compute_format[computeID]:
                    _compute_export += [lmp.numpy.extract_compute (f"ave_{computeID}_{_rid}", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)]
                else:
                    _compute_export += [[lmp.numpy.extract_compute (f"ave_{computeID}_{_rid}", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR)]]

                if _nvi == 0: # use first voxel to understand number of compute entries for the header 
                    compute_header += f"{computeID}: {self.compute_string[computeID]} ({len(_compute_export[-1])}), "

            compute_export = np.hstack([compute_export, _compute_export])

        compute_header = "# %s\n" % compute_header.rstrip(", ")

        if (self.me == exporting_rank):
            with open(filepath, 'w') as fopen:
                fopen.write(f"# FRAME ID\n")
                fopen.write(f"{frameID}\n")

                fopen.write("\n# NVOXELS NCOMPUTES\n")
                fopen.write("%d %d\n" % compute_export.shape)

                fopen.write("\n# xlo xhi ylo yhi zlo zhi xy xz yz\n")
                for _row in voxelbounds_export:
                    fopen.write("%lf "*len(_row) % tuple(_row) + "\n")

                fopen.write("\n# COMPUTE STRING\n")
                fopen.write(compute_header)

                fopen.write("\n# COMPUTES\n")
                for _row in compute_export:
                    fopen.write("%f "*len(_row) % tuple(_row) + "\n")


class Voxel_Descriptors (_Voxel):
    def __init__(self, all_input, *args, **kwargs):

        # first initialise standard voxelisation
        super().__init__(*args, **kwargs)

        lmp = self.lmp 

        # then define computes for time-averaged unwrapped coordinates
        # to avoid issues with averaging positions across PBC, use unwrapped coordinates
        export_descriptors_nth = all_input['export_descriptors_nth']
        ave_steps = int(1e-3*all_input["average_over_fs"]/all_input["timestep"])
        lmp.command('compute cunw all property/atom xu yu zu')
        lmp.command('compute cimg all property/atom ix iy iz')
        lmp.command(f"fix ave all ave/atom 1 {ave_steps} {export_descriptors_nth} c_cunw[*]")
        lmp.command("variable ax atom f_ave[1]")
        lmp.command("variable ay atom f_ave[2]")
        lmp.command("variable az atom f_ave[3]")

        lmp.command("variable aix atom c_cimg[1]")
        lmp.command("variable aiy atom c_cimg[2]")
        lmp.command("variable aiz atom c_cimg[3]")

        # store important values for descriptor evaluation as properties of this class
        self.compute_string = all_input["descriptor_computes"]
        self.compute_format = {key:"[*]" for key in self.compute_string}
        self.export_trajectories_nth = all_input["export_trajectories_nth"]
        self.export_descriptors_nth = export_descriptors_nth
        self.ave_steps = ave_steps


    def set_atoms_to_averaged (self):
        lmp = self.lmp

        # fetch momentary atomic coordinates 
        nlocal = lmp.extract_global('nlocal', dtype=LAMMPS_INT)
        xyz_rank = lmp.numpy.extract_atom ("x")
 
        # fetch time-averaged unwrapped atomic coordinates
        uxyz_rank = lmp.numpy.extract_fix ("ave", LMP_STYLE_ATOM, LMP_TYPE_ARRAY)

        # fetch momentary image indices 
        ix_rank = lmp.numpy.extract_variable ("aix", vartype=LMP_VAR_ATOM)
        iy_rank = lmp.numpy.extract_variable ("aiy", vartype=LMP_VAR_ATOM)
        iz_rank = lmp.numpy.extract_variable ("aiz", vartype=LMP_VAR_ATOM)
        img_rank = np.c_[ix_rank, iy_rank, iz_rank]

        # transform time-averaged unwrapped atomic coordinates into periodic image of the momentary atoms 
        # this is to avoid a 'lost atoms' error as unwrapped atoms can be in a different periodic image 
        cell = getcell(lmp)
        affine = AffineTransform(cell, cell)
        uxyz_rank = (uxyz_rank - np.einsum('ik,jk->ij', img_rank[:nlocal], affine.c1mat))

        # now that variables are current, and averaged trajectories are available, compute the descriptor
        lmp.command("fix storage all store/state 0 x y z")
   
        # set to ave positions
        xyz_rank_ptr = lmp.extract_atom("x")
        for ni in range(nlocal):
            xyz_rank_ptr[ni][0] = uxyz_rank[ni][0]
            xyz_rank_ptr[ni][1] = uxyz_rank[ni][1]
            xyz_rank_ptr[ni][2] = uxyz_rank[ni][2]


    def return_atoms_from_averaged (self): 
        lmp = self.lmp

        # set back to momentary positions 
        lmp.command("variable x atom f_storage[1]")
        lmp.command("variable y atom f_storage[2]")
        lmp.command("variable z atom f_storage[3]")
        lmp.command("set atom * x v_x")
        lmp.command("set atom * y v_y")
        lmp.command("set atom * z v_z")

        lmp.command("unfix storage")
        lmp.command("run 0")

