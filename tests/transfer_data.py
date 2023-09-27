# -*- coding: utf-8 -*-

from mpi4py import MPI
import os

import jax
import jax.numpy as jnp

from jaxnvtransfer import nvtransfer

def main():
    comm = MPI.COMM_WORLD
    nrank = comm.Get_rank()
    nsize = comm.Get_size()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (nrank+2)
    os.environ["JAX_ENABLE_X64"] = "True"

    send_size = 101 * (nrank + 2)
    # dtype = jnp.float32
    dtype = jnp.float64
    key = jax.random.PRNGKey(nrank)
    send_data = jax.random.normal(key, shape=(send_size,), dtype=dtype)
    # input = jnp.ones(shape=input_shape, dtype=dtype)

    dst_rank = nrank + 1
    dst_rank = dst_rank % nsize

    src_rank = nrank - 1
    if(src_rank < 0):
        src_rank = nsize - 1

    # print(f'rank is {nrank}, send size is {send_size}, dst rank is {dst_rank}')
    comm.isend(send_size, dst_rank, tag=11)
    req = comm.irecv(source=src_rank, tag=11)
    recv_size = req.wait()
    # print(f'rank is {nrank}, recv size is {recv_size}, src rank is {src_rank}')

    # nvtransfer(send_buf, send_buf_size, recv_buf_size, src_rank, dst_rank)
    recv_data = nvtransfer(send_data, send_size, recv_size, src_rank, dst_rank)

    if(nrank == 0):
        print(f'send size = {send_size}')
        print(send_data)
    if(nrank == 1):
        print(f'recv size = {recv_size}')
        print(recv_data)

if __name__ == "__main__":
    main()
    # NVSHMEM_BOOTSTRAP=MPI mpirun --allow-run-as-root -n 2 python tests/transfer_data.py

