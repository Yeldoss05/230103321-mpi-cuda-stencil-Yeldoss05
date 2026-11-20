import os
import sys
import json
import math
import time
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
SUB_NX = 256
SUB_NY = 256
LOC_NX = SUB_NX + 2  # 258
LOC_NY = SUB_NY + 2  # 258
STEPS = 400
LAMBDA_DAMP = 0.15


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    cart_rank = cart_comm.Get_rank()
    row_rank, col_rank = cart_comm.Get_coords(cart_rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    return {
        "world_comm": comm,
        "cart_comm": cart_comm,
        "rank": cart_rank,
        "size": size,
        "row_rank": row_rank,
        "col_rank": col_rank,
        "nbr_up": nbr_up,
        "nbr_down": nbr_down,
        "nbr_left": nbr_left,
        "nbr_right": nbr_right,
        "gpu_uuid": gpu_uuid,
        "instance_seed": instance_seed,
    }


def initialize_subdomains(ctx):
    row_rank = ctx["row_rank"]
    col_rank = ctx["col_rank"]
    nbr_up = ctx["nbr_up"]
    nbr_down = ctx["nbr_down"]
    nbr_left = ctx["nbr_left"]
    nbr_right = ctx["nbr_right"]
    seed = ctx["instance_seed"]

    # Grid spacing across 513 nodes so interior has 512x512 cells on [0, 1]^2
    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    # Global 0-based cell index for local index i in 0..257:
    # interior local 1..256 maps to global 0..255 (shifted by row_rank * 256)
    g_rows = row_rank * SUB_NY + (np.arange(LOC_NY, dtype=np.float64) - 0.5)
    g_cols = col_rank * SUB_NX + (np.arange(LOC_NX, dtype=np.float64) - 0.5)
    x_1d = np.clip((g_cols + 1.0) * dx, 0.0, 1.0)
    y_1d = np.clip((g_rows + 1.0) * dy, 0.0, 1.0)
    X, Y = np.meshgrid(x_1d, y_1d)  # shape (258, 258)

    rng = np.random.default_rng(seed)
    c1_x, c1_y = rng.uniform(0.25, 0.75, size=2)
    a1 = rng.uniform(0.6, 1.0)
    c2_x, c2_y = rng.uniform(0.25, 0.75, size=2)
    a2 = rng.uniform(0.4, 0.8)

    k_host = 0.08 + 0.16 * (
        0.5 + 0.5 * np.sin(2.0 * math.pi * X) * np.cos(2.0 * math.pi * Y)
    )
    k_host = np.clip(k_host, 0.08, 0.24).astype(np.float64)

    u_host = (
        a1 * np.exp(-((X - c1_x) ** 2 + (Y - c1_y) ** 2) / (2.0 * 0.08 ** 2))
        + a2 * np.exp(-((X - c2_x) ** 2 + (Y - c2_y) ** 2) / (2.0 * 0.10 ** 2))
    ).astype(np.float64)

    s_host = (
        0.05 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2.0 * 0.12 ** 2))
    ).astype(np.float64)

    # Enforce fixed homogeneous Dirichlet u = 0.0 on outer global boundaries
    if nbr_up == MPI.PROC_NULL:
        u_host[0, :] = 0.0
    if nbr_down == MPI.PROC_NULL:
        u_host[-1, :] = 0.0
    if nbr_left == MPI.PROC_NULL:
        u_host[:, 0] = 0.0
    if nbr_right == MPI.PROC_NULL:
        u_host[:, -1] = 0.0

    k_max = float(np.max(k_host[1:-1, 1:-1]))
    dt_max = (dx * dx) / (4.0 * k_max)
    dt = 0.40 * dt_max

    u_curr_d = cuda.to_device(u_host)
    u_next_d = cuda.to_device(u_host.copy())
    k_d = cuda.to_device(k_host)
    s_d = cuda.to_device(s_host)

    return {
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "dt_max": dt_max,
        "k_max": k_max,
        "u_curr_d": u_curr_d,
        "u_next_d": u_next_d,
        "k_d": k_d,
        "s_d": s_d,
        "u_init_host": u_host.copy(),
    }


@cuda.jit
def pack_halos_kernel(u, send_up, send_down, send_left, send_right, ny, nx):
    idx = cuda.grid(1)
    if idx < nx:
        send_up[idx] = u[1, idx + 1]
        send_down[idx] = u[ny, idx + 1]
    if idx < ny:
        send_left[idx] = u[idx + 1, 1]
        send_right[idx] = u[idx + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, recv_up, recv_down, recv_left, recv_right, ny, nx):
    idx = cuda.grid(1)
    if idx < nx:
        u[0, idx + 1] = recv_up[idx]
        u[ny + 1, idx + 1] = recv_down[idx]
    if idx < ny:
        u[idx + 1, 0] = recv_left[idx]
        u[idx + 1, nx + 1] = recv_right[idx]


@cuda.jit
def tiled_stencil_kernel(
    u_curr, u_next, k_field, s_field,
    ny, nx, dx, dy, dt, lam,
    r_start, r_end, c_start, c_end
):
    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    k_smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Local array coordinates (0..257) for block origin
    base_r = by * TILE_DIM
    base_c = bx * TILE_DIM
    r = base_r + ty
    c = base_c + tx

    sy = ty + 1
    sx = tx + 1

    # Load center cell into shared memory
    if r < ny + 2 and c < nx + 2:
        smem[sy, sx] = u_curr[r, c]
        k_smem[sy, sx] = k_field[r, c]
    else:
        smem[sy, sx] = 0.0
        k_smem[sy, sx] = 0.0

    # Load 1-cell halo apron around the 16x16 tile
    if ty == 0:
        r_up = base_r - 1
        if r_up >= 0 and c < nx + 2:
            smem[0, sx] = u_curr[r_up, c]
            k_smem[0, sx] = k_field[r_up, c]
        else:
            smem[0, sx] = 0.0
            k_smem[0, sx] = 0.0

    if ty == TILE_DIM - 1 or r == ny:
        r_dn = base_r + TILE_DIM
        if r_dn < ny + 2 and c < nx + 2:
            smem[SHARED_DIM - 1, sx] = u_curr[r_dn, c]
            k_smem[SHARED_DIM - 1, sx] = k_field[r_dn, c]
        else:
            smem[SHARED_DIM - 1, sx] = 0.0
            k_smem[SHARED_DIM - 1, sx] = 0.0

    if tx == 0:
        c_lf = base_c - 1
        if r < ny + 2 and c_lf >= 0:
            smem[sy, 0] = u_curr[r, c_lf]
            k_smem[sy, 0] = k_field[r, c_lf]
        else:
            smem[sy, 0] = 0.0
            k_smem[sy, 0] = 0.0

    if tx == TILE_DIM - 1 or c == nx:
        c_rt = base_c + TILE_DIM
        if r < ny + 2 and c_rt < nx + 2:
            smem[sy, SHARED_DIM - 1] = u_curr[r, c_rt]
            k_smem[sy, SHARED_DIM - 1] = k_field[r, c_rt]
        else:
            smem[sy, SHARED_DIM - 1] = 0.0
            k_smem[sy, SHARED_DIM - 1] = 0.0

    # Corner apron cells
    if ty == 0 and tx == 0:
        r_c = base_r - 1
        c_c = base_c - 1
        smem[0, 0] = u_curr[r_c, c_c] if (r_c >= 0 and c_c >= 0) else 0.0
        k_smem[0, 0] = k_field[r_c, c_c] if (r_c >= 0 and c_c >= 0) else 0.0

    if ty == 0 and tx == TILE_DIM - 1:
        r_c = base_r - 1
        c_c = base_c + TILE_DIM
        smem[0, SHARED_DIM - 1] = (
            u_curr[r_c, c_c] if (r_c >= 0 and c_c < nx + 2) else 0.0
        )
        k_smem[0, SHARED_DIM - 1] = (
            k_field[r_c, c_c] if (r_c >= 0 and c_c < nx + 2) else 0.0
        )

    if ty == TILE_DIM - 1 and tx == 0:
        r_c = base_r + TILE_DIM
        c_c = base_c - 1
        smem[SHARED_DIM - 1, 0] = (
            u_curr[r_c, c_c] if (r_c < ny + 2 and c_c >= 0) else 0.0
        )
        k_smem[SHARED_DIM - 1, 0] = (
            k_field[r_c, c_c] if (r_c < ny + 2 and c_c >= 0) else 0.0
        )

    if ty == TILE_DIM - 1 and tx == TILE_DIM - 1:
        r_c = base_r + TILE_DIM
        c_c = base_c + TILE_DIM
        smem[SHARED_DIM - 1, SHARED_DIM - 1] = (
            u_curr[r_c, c_c] if (r_c < ny + 2 and c_c < nx + 2) else 0.0
        )
        k_smem[SHARED_DIM - 1, SHARED_DIM - 1] = (
            k_field[r_c, c_c] if (r_c < ny + 2 and c_c < nx + 2) else 0.0
        )

    cuda.syncthreads()

    if r_start <= r < r_end and c_start <= c < c_end:
        u_c = smem[sy, sx]
        u_n = smem[sy - 1, sx]
        u_s = smem[sy + 1, sx]
        u_w = smem[sy, sx - 1]
        u_e = smem[sy, sx + 1]

        k_c = k_smem[sy, sx]
        k_n = k_smem[sy - 1, sx]
        k_s = k_smem[sy + 1, sx]
        k_w = k_smem[sy, sx - 1]
        k_e = k_smem[sy, sx + 1]

        k_iph = 0.5 * (k_e + k_c)
        k_imh = 0.5 * (k_c + k_w)
        k_jph = 0.5 * (k_s + k_c)
        k_jmh = 0.5 * (k_c + k_n)

        diff_x = (k_iph * (u_e - u_c) - k_imh * (u_c - u_w)) / (dx * dx)
        diff_y = (k_jph * (u_s - u_c) - k_jmh * (u_c - u_n)) / (dy * dy)
        react = -lam * (u_c * u_c * u_c) + s_field[r, c]

        u_next[r, c] = u_c + dt * (diff_x + diff_y + react)


class HaloBuffers:
    def __init__(self):
        self.s_up_d = cuda.device_array(SUB_NX, dtype=np.float64)
        self.s_dn_d = cuda.device_array(SUB_NX, dtype=np.float64)
        self.s_lf_d = cuda.device_array(SUB_NY, dtype=np.float64)
        self.s_rt_d = cuda.device_array(SUB_NY, dtype=np.float64)

        self.r_up_d = cuda.device_array(SUB_NX, dtype=np.float64)
        self.r_dn_d = cuda.device_array(SUB_NX, dtype=np.float64)
        self.r_lf_d = cuda.device_array(SUB_NY, dtype=np.float64)
        self.r_rt_d = cuda.device_array(SUB_NY, dtype=np.float64)

        self.s_up_h = cuda.pinned_array(SUB_NX, dtype=np.float64)
        self.s_dn_h = cuda.pinned_array(SUB_NX, dtype=np.float64)
        self.s_lf_h = cuda.pinned_array(SUB_NY, dtype=np.float64)
        self.s_rt_h = cuda.pinned_array(SUB_NY, dtype=np.float64)

        self.r_up_h = cuda.pinned_array(SUB_NX, dtype=np.float64)
        self.r_dn_h = cuda.pinned_array(SUB_NX, dtype=np.float64)
        self.r_lf_h = cuda.pinned_array(SUB_NY, dtype=np.float64)
        self.r_rt_h = cuda.pinned_array(SUB_NY, dtype=np.float64)


def post_halo_exchange(ctx, bufs: HaloBuffers):
    cart_comm = ctx["cart_comm"]
    reqs = []
    reqs.append(cart_comm.Irecv(bufs.r_up_h, source=ctx["nbr_up"], tag=10))
    reqs.append(cart_comm.Irecv(bufs.r_dn_h, source=ctx["nbr_down"], tag=11))
    reqs.append(cart_comm.Irecv(bufs.r_lf_h, source=ctx["nbr_left"], tag=12))
    reqs.append(cart_comm.Irecv(bufs.r_rt_h, source=ctx["nbr_right"], tag=13))

    reqs.append(cart_comm.Isend(bufs.s_up_h, dest=ctx["nbr_up"], tag=11))
    reqs.append(cart_comm.Isend(bufs.s_dn_h, dest=ctx["nbr_down"], tag=10))
    reqs.append(cart_comm.Isend(bufs.s_lf_h, dest=ctx["nbr_left"], tag=13))
    reqs.append(cart_comm.Isend(bufs.s_rt_h, dest=ctx["nbr_right"], tag=12))
    return reqs


def launch_boundary_ring_stencils(
    u_curr_d, u_next_d, k_d, s_d, dx, dy, dt, stream
):
    grid_2d = (
        math.ceil(LOC_NX / TILE_DIM),
        math.ceil(LOC_NY / TILE_DIM),
    )
    block_2d = (TILE_DIM, TILE_DIM)

    # Top ring: rows [1, 2), cols [1, 257)
    tiled_stencil_kernel[grid_2d, block_2d, stream](
        u_curr_d, u_next_d, k_d, s_d,
        SUB_NY, SUB_NX, dx, dy, dt, LAMBDA_DAMP,
        1, 2, 1, SUB_NX + 1
    )
    # Bottom ring: rows [256, 257), cols [1, 257)
    tiled_stencil_kernel[grid_2d, block_2d, stream](
        u_curr_d, u_next_d, k_d, s_d,
        SUB_NY, SUB_NX, dx, dy, dt, LAMBDA_DAMP,
        SUB_NY, SUB_NY + 1, 1, SUB_NX + 1
    )
    # Left ring: rows [2, 256), cols [1, 2)
    tiled_stencil_kernel[grid_2d, block_2d, stream](
        u_curr_d, u_next_d, k_d, s_d,
        SUB_NY, SUB_NX, dx, dy, dt, LAMBDA_DAMP,
        2, SUB_NY, 1, 2
    )
    # Right ring: rows [2, 256), cols [256, 257)
    tiled_stencil_kernel[grid_2d, block_2d, stream](
        u_curr_d, u_next_d, k_d, s_d,
        SUB_NY, SUB_NX, dx, dy, dt, LAMBDA_DAMP,
        2, SUB_NY, SUB_NX, SUB_NX + 1
    )


def step_serialized(ctx, state, bufs: HaloBuffers, stream_b):
    u_curr_d = state["u_curr_d"]
    u_next_d = state["u_next_d"]
    k_d = state["k_d"]
    s_d = state["s_d"]
    dx, dy, dt = state["dx"], state["dy"], state["dt"]

    threads_1d = 256
    blocks_1d = math.ceil(max(SUB_NX, SUB_NY) / threads_1d)
    pack_halos_kernel[blocks_1d, threads_1d, stream_b](
        u_curr_d, bufs.s_up_d, bufs.s_dn_d, bufs.s_lf_d, bufs.s_rt_d, SUB_NY, SUB_NX
    )
    bufs.s_up_d.copy_to_host(bufs.s_up_h, stream=stream_b)
    bufs.s_dn_d.copy_to_host(bufs.s_dn_h, stream=stream_b)
    bufs.s_lf_d.copy_to_host(bufs.s_lf_h, stream=stream_b)
    bufs.s_rt_d.copy_to_host(bufs.s_rt_h, stream=stream_b)
    stream_b.synchronize()

    reqs = post_halo_exchange(ctx, bufs)
    MPI.Request.Waitall(reqs)

    bufs.r_up_d.copy_to_device(bufs.r_up_h, stream=stream_b)
    bufs.r_dn_d.copy_to_device(bufs.r_dn_h, stream=stream_b)
    bufs.r_lf_d.copy_to_device(bufs.r_lf_h, stream=stream_b)
    bufs.r_rt_d.copy_to_device(bufs.r_rt_h, stream=stream_b)
    unpack_halos_kernel[blocks_1d, threads_1d, stream_b](
        u_curr_d, bufs.r_up_d, bufs.r_dn_d, bufs.r_lf_d, bufs.r_rt_d, SUB_NY, SUB_NX
    )
    stream_b.synchronize()

    grid_2d = (math.ceil(LOC_NX / TILE_DIM), math.ceil(LOC_NY / TILE_DIM))
    block_2d = (TILE_DIM, TILE_DIM)
    tiled_stencil_kernel[grid_2d, block_2d, stream_b](
        u_curr_d, u_next_d, k_d, s_d,
        SUB_NY, SUB_NX, dx, dy, dt, LAMBDA_DAMP,
        1, SUB_NY + 1, 1, SUB_NX + 1
    )
    stream_b.synchronize()

    state["u_curr_d"], state["u_next_d"] = u_next_d, u_curr_d


def step_overlapped(ctx, state, bufs: HaloBuffers, stream_b, stream_i):
    u_curr_d = state["u_curr_d"]
    u_next_d = state["u_next_d"]
    k_d = state["k_d"]
    s_d = state["s_d"]
    dx, dy, dt = state["dx"], state["dy"], state["dt"]

    threads_1d = 256
    blocks_1d = math.ceil(max(SUB_NX, SUB_NY) / threads_1d)
    grid_2d = (math.ceil(LOC_NX / TILE_DIM), math.ceil(LOC_NY / TILE_DIM))
    block_2d = (TILE_DIM, TILE_DIM)

    # Stage A: Pack boundary halos & async D2H on stream_boundary
    pack_halos_kernel[blocks_1d, threads_1d, stream_b](
        u_curr_d, bufs.s_up_d, bufs.s_dn_d, bufs.s_lf_d, bufs.s_rt_d, SUB_NY, SUB_NX
    )
    bufs.s_up_d.copy_to_host(bufs.s_up_h, stream=stream_b)
    bufs.s_dn_d.copy_to_host(bufs.s_dn_h, stream=stream_b)
    bufs.s_lf_d.copy_to_host(bufs.s_lf_h, stream=stream_b)
    bufs.s_rt_d.copy_to_host(bufs.s_rt_h, stream=stream_b)

    # Stage B: Sync boundary stream, post non-blocking MPI, compute interior [2..255, 2..255] on stream_interior
    stream_b.synchronize()
    reqs = post_halo_exchange(ctx, bufs)
    tiled_stencil_kernel[grid_2d, block_2d, stream_i](
        u_curr_d, u_next_d, k_d, s_d,
        SUB_NY, SUB_NX, dx, dy, dt, LAMBDA_DAMP,
        2, SUB_NY, 2, SUB_NX
    )

    # Stage C: Wait MPI, async H2D, unpack halos, compute 4 outer rings on stream_boundary
    MPI.Request.Waitall(reqs)
    bufs.r_up_d.copy_to_device(bufs.r_up_h, stream=stream_b)
    bufs.r_dn_d.copy_to_device(bufs.r_dn_h, stream=stream_b)
    bufs.r_lf_d.copy_to_device(bufs.r_lf_h, stream=stream_b)
    bufs.r_rt_d.copy_to_device(bufs.r_rt_h, stream=stream_b)
    unpack_halos_kernel[blocks_1d, threads_1d, stream_b](
        u_curr_d, bufs.r_up_d, bufs.r_dn_d, bufs.r_lf_d, bufs.r_rt_d, SUB_NY, SUB_NX
    )
    launch_boundary_ring_stencils(
        u_curr_d, u_next_d, k_d, s_d, dx, dy, dt, stream_b
    )

    stream_b.synchronize()
    stream_i.synchronize()

    state["u_curr_d"], state["u_next_d"] = u_next_d, u_curr_d


def compute_global_diagnostics(cart_comm, u_curr_d, u_prev_d, dx, dy, dt):
    u_curr = u_curr_d.copy_to_host()[1:-1, 1:-1]
    u_prev = u_prev_d.copy_to_host()[1:-1, 1:-1]

    local_energy = float(np.sum(u_curr) * dx * dy)
    local_diff_sq = float(np.sum(((u_curr - u_prev) / dt) ** 2) * dx * dy)

    global_energy = cart_comm.allreduce(local_energy, op=MPI.SUM)
    global_l2_sq = cart_comm.allreduce(local_diff_sq, op=MPI.SUM)
    return float(global_energy), float(math.sqrt(global_l2_sq))


def audit_ptx():
    ptx_dict = tiled_stencil_kernel.inspect_asm()
    ptx_text = "\n".join(str(v) for v in ptx_dict.values()) if isinstance(ptx_dict, dict) else str(ptx_dict)
    st_shared_count = ptx_text.count("st.shared")
    ld_shared_count = ptx_text.count("ld.shared")
    bar_sync_count = ptx_text.count("bar.sync")
    passed = (st_shared_count > 0) and (ld_shared_count > 0) and (bar_sync_count > 0)
    return {
        "passed": passed,
        "st_shared_count": st_shared_count,
        "ld_shared_count": ld_shared_count,
        "bar_sync_count": bar_sync_count,
    }


def run_solver(github_user: str):
    ctx = init_distributed_context(github_user)
    cart_comm = ctx["cart_comm"]
    rank = ctx["rank"]

    bufs = HaloBuffers()
    stream_b = cuda.stream()
    stream_i = cuda.stream()

    # 1. Serialized run (400 steps)
    state_ser = initialize_subdomains(ctx)
    energy_trace = []
    e0, l2_0 = compute_global_diagnostics(
        cart_comm, state_ser["u_curr_d"], state_ser["u_curr_d"],
        state_ser["dx"], state_ser["dy"], state_ser["dt"]
    )
    energy_trace.append({"step": 0, "energy": e0, "l2_residual": l2_0})

    t0_ser = time.perf_counter()
    for step in range(1, STEPS + 1):
        step_serialized(ctx, state_ser, bufs, stream_b)
        if step % 100 == 0:
            en, l2 = compute_global_diagnostics(
                cart_comm, state_ser["u_curr_d"], state_ser["u_next_d"],
                state_ser["dx"], state_ser["dy"], state_ser["dt"]
            )
            energy_trace.append({"step": step, "energy": en, "l2_residual": l2})
    serialized_ms = (time.perf_counter() - t0_ser) * 1000.0

    # 2. Overlapped run (400 steps)
    state_ov = initialize_subdomains(ctx)
    t0_ov = time.perf_counter()
    for _ in range(1, STEPS + 1):
        step_overlapped(ctx, state_ov, bufs, stream_b, stream_i)
    overlapped_ms = (time.perf_counter() - t0_ov) * 1000.0

    final_ser = state_ser["u_curr_d"].copy_to_host()
    final_ov = state_ov["u_curr_d"].copy_to_host()
    local_max_diff = float(np.max(np.abs(final_ser - final_ov)))
    parity_max_diff = float(cart_comm.allreduce(local_max_diff, op=MPI.MAX))
    assert parity_max_diff < 1e-12, f"Parity check failed: {parity_max_diff}"

    ptx_audit = audit_ptx() if rank == 0 else None
    ptx_audit = cart_comm.bcast(ptx_audit, root=0)

    if rank == 0:
        print(f"GITHUB_USERNAME: {github_user}")
        print(f"MPI ranks: {ctx['size']} (2x2 grid), GPU UUID: {ctx['gpu_uuid']}")
        print(f"INSTANCE_SEED: {ctx['instance_seed']}")
        print(
            f"k_max={state_ser['k_max']:.6f}, dx={state_ser['dx']:.8f}, "
            f"dt_max={state_ser['dt_max']:.8e}, dt={state_ser['dt']:.8e}"
        )
        print(f"serialized_ms: {serialized_ms:.2f} ms")
        print(f"overlapped_ms: {overlapped_ms:.2f} ms")
        print(f"parity_max_diff: {parity_max_diff:.3e}")
        print(f"PTX audit: {ptx_audit}")
        print("Energy trace (steps 0, 100, 200, 300, 400):")
        for row in energy_trace:
            print(
                f"  step={row['step']:3d} | energy={row['energy']:.8e} | "
                f"l2_residual={row['l2_residual']:.8e}"
            )

        manifest = {
            "GITHUB_USERNAME": github_user,
            "gpu_uuid": ctx["gpu_uuid"],
            "instance_seed": str(ctx["instance_seed"]),
            "mpi_ranks": ctx["size"],
            "grid_dims": [2, 2],
            "steps": STEPS,
            "k_max": state_ser["k_max"],
            "dx": state_ser["dx"],
            "dt_max": state_ser["dt_max"],
            "dt": state_ser["dt"],
            "serialized_ms": serialized_ms,
            "overlapped_ms": overlapped_ms,
            "parity_max_diff": parity_max_diff,
            "ptx_audit_passed": ptx_audit["passed"],
            "ptx_audit": ptx_audit,
            "energy_trace": energy_trace,
        }
        with open("submission_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Saved submission_manifest.json")


if __name__ == "__main__":
    user_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_USERNAME", "")
    run_solver(user_arg)
