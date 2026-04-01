import sys
from lammps import lammps

# template to replace MPI functionality for single threaded use
class MPI_to_serial():
    def bcast(self, *args, **kwargs):
        return args[0]
    def barrier(self):
        return 0

# try running in parallel, otherwise single thread
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    me = comm.Get_rank()
    nprocs = comm.Get_size()
    mode = 'MPI'
except:
    me = 0
    nprocs = 1
    comm = MPI_to_serial()
    mode = 'serial'

def mpiprint(*arg):
    if me == 0:
        print(*arg)
        sys.stdout.flush()
    return 0


def main():

    sys.stdout.flush()
    print ("From Python: I am rank ID %d out of %d ranks." % (me, nprocs))
    sys.stdout.flush()

    lmp = lammps("gpu-cuda")
    lmp.command("print 'From LAMMPS: Hello World. You should see this message only once.'")
    lmp.close()

    return 0


if __name__ == "__main__":
    main()

    if mode == 'MPI':
        MPI.Finalize()
