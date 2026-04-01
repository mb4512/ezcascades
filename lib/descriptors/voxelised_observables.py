import numpy as np

from lib.descriptors.voxelised_descriptors import _Voxel

class Voxel_Observables (_Voxel):
    def __init__(self, all_input, *args, **kwargs):

        # first initialise standard voxelisation
        super().__init__(*args, **kwargs)

        lmp = self.lmp 

        # store important values for descriptor evaluation as properties of this class
        self.compute_string = all_input["observable_computes"]
        self.compute_format = all_input["observable_format"] 
