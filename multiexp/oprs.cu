#pragma once
#include <cstdint>
#include <vector>
#include <chrono>
#include <memory>
#include <cooperative_groups.h>

#include "curves.cu"
#include "loader.cu"
template <typename EC>
__global__ void
ec_cpy_jac2jac_kernel(var *dst, const var *src, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n)
    {
        EC x;
        int off = idx * EC::NELTS * ELT_LIMBS;
        EC::load_jac(x, src + off);
        EC::store_jac(dst + off, x);
    }
}

template <typename EC>
__global__ void
ec_cpy_aff2jac_kernel(var *dst, const var *src, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n)
    {
        EC x;
        int o_off = idx * 3 * EC::field_type::DEGREE * ELT_LIMBS;
        int x_off = idx * 2 * EC::field_type::DEGREE * ELT_LIMBS;
        EC::load_affine(x, src + x_off);
        EC::store_jac(dst + o_off, x);
    }
}

template <typename EC>
__global__ void ec_setZ_jac_kernel(var *mem, size_t N)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;
    if (idx < N)
    {
        EC x;
        EC::set_zero(x);
        EC::store_jac(mem + idx * 3 * EC::field_type::DEGREE * ELT_LIMBS, x);
    }
}

template <typename EC, int EXP>
__global__ void
ec_mul_2exp_kernel(var *dst, const var *src, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n)
    {
        EC x;
        int off = idx * EC::NELTS * ELT_LIMBS;
        EC::load_jac(x, src + off);
        for (int k = 0; k < EXP; k++)
        {
            EC::dbl(x, x);
        }
        EC::store_jac(dst + off, x);
    }
}

template <typename EC>
__global__ void
ec_memcpy_withcheck(var *dst, const var *src, const var *scalars, const int pos, size_t n)
{
    // when  scalar[idx][pos]==0 set dst to zero
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;
    typedef typename EC::group_type Fr;
    if (idx < n)
    {
        EC x;
        int off = idx * EC::NELTS * ELT_LIMBS;
        int s_off = idx * ELT_LIMBS;
        Fr s;
        Fr::load(s, scalars + s_off + idx * ELT_LIMBS);
        Fr::from_monty(s, s);
        int q = pos / digit::BITS, r = pos % digit::BITS;
        auto g = fixnum::layout();
        var ss = g.shfl(s.a, q);
        var win = (ss >> r) & 1U;
        if (win > 0)
        {
            EC::load_jac(x, src + off);
        }
        else
        {
            EC::set_zero(x);
        }
        EC::store_jac(dst + off, x);
    }
}

template <typename EC>
__global__ void
ec_sum_all_kernel(var *X, const var *Y, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n)
    {
        EC z, x, y;
        int off = idx * EC::NELTS * ELT_LIMBS;

        EC::load_jac(x, X + off);
        EC::load_jac(y, Y + off);

        EC::add(z, x, y);

        EC::store_jac(X + off, z);
    }
}

template <typename EC>
__global__ void
fr_from_monty_kernel(var *dst, const var *src, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n)
    {
        typedef typename EC::group_type Fr;
        Fr w;
        int w_off = idx * ELT_LIMBS;
        Fr::load(w, src + w_off);
        Fr::from_monty(w, w);
        Fr::store(dst + w_off, w);
    }
}

template <typename EC>
void ec_cpy_jac2jac_async(cudaStream_t &strm, var *dst, const var *src, size_t n)
{
    static constexpr size_t TPB = 256;
    size_t bN = (n * BIG_WIDTH + TPB - 1) / TPB;
    ec_cpy_jac2jac_kernel<EC><<<bN, TPB, 0, strm>>>(dst, src, n);
}

template <typename EC>
void ec_cpy_aff2jac_async(cudaStream_t &strm, var *dst, const var *src, size_t n)
{
    static constexpr size_t TPB = 256;
    size_t bN = (n * BIG_WIDTH + TPB - 1) / TPB;
    ec_cpy_aff2jac_kernel<EC><<<bN, TPB, 0, strm>>>(dst, src, n);
}

template <typename EC>
void ec_setZ_jac_async(cudaStream_t &strm, var *dst, size_t n)
{
    static constexpr size_t TPB = 256;
    size_t bN = (n * BIG_WIDTH + TPB - 1) / TPB;
    ec_setZ_jac_kernel<EC><<<bN, TPB, 0, strm>>>(dst, n);
}

template <typename EC, int EXP>
void ec_mul_2exp_jac_async(cudaStream_t &strm, var *dst, var *multiples, size_t n)
{
    static constexpr size_t TPB = 256;
    size_t bN = (n * BIG_WIDTH + TPB - 1) / TPB;
    ec_mul_2exp_kernel<EC, EXP><<<bN, TPB, 0, strm>>>(dst, multiples, n);
}

template <typename EC>
void ec_sum_all_jac_async(cudaStream_t &strm, var *X, const var *Y, size_t n)
{
    static constexpr size_t TPB = 256;
    size_t bN = (n * BIG_WIDTH + TPB - 1) / TPB;
    ec_sum_all_kernel<EC><<<bN, TPB, 0, strm>>>(X, Y, n);
}

template <typename EC>
void fr_from_monty(cudaStream_t &strm, var *dst, const var *src, size_t n)
{
    static constexpr size_t TPB = 256;
    size_t bN = (n * BIG_WIDTH + TPB - 1) / TPB;
    fr_from_monty_kernel<EC><<<bN, TPB, 0, strm>>>(dst, src, n);
}


template <typename EC>
void assert_equal_jac(std::string msg,const var *x, const var *y, size_t n)
{
    for (int i = 0; i < n * EC::NELTS * ELT_LIMBS; i++)
    {
        if(x[i]!=y[i]){
            printf("Assert Failed!%s\n",msg.c_str());
            exit(1);
        }
        //assert(x[i] != y[i]);
    }
    printf("Assert Passed.%s\n",msg.c_str());
}

template <typename EC>
void ec_sum_reduce(cudaStream_t &strm, var *out, size_t n)
{
    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;
    static constexpr size_t threads_per_block = 256;
    size_t nblocks = (n * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    size_t r = n & 1, m = n / 2;
    for (; m != 0; r = m & 1, m >>= 1)
    {
        nblocks = (m * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

        ec_sum_all_kernel<EC><<<nblocks, threads_per_block, 0, strm>>>(out, out + m * pt_limbs, m);
        if (r)
            ec_sum_all_kernel<EC><<<1, threads_per_block, 0, strm>>>(out, out + 2 * m * pt_limbs, 1);
    }
}
/*
void opr_unit_test()
{
    const char *preprocessed_path = "MNT4753_preprocessed";
    const char *input_path = "MNT6753-input";
    const char *params_path = "MNT6753-parameters";
    printf("Operations Unit Test.\n");
    typedef mnt6753_libsnark ppT;
    typedef ECp_MNT6 ECp;
    typedef ECp3_MNT6 ECpe;

    typedef typename ppT::G1 G1;
    typedef typename ppT::G2 G2;

    ppT::init_public_params();

    size_t primary_input_size = 1;

    FILE *params_file = fopen(params_path, "r");
    size_t d = read_size_t(params_file);
    size_t m = read_size_t(params_file);
    rewind(params_file);

    printf("d = %zu, m = %zu\n", d, m);

    static constexpr int R = 32;
    static constexpr int C = 5;
    FILE *preprocessed_file = fopen(preprocessed_path, "r");
    auto B1_mults = load_points_affine<ECp>(((1U << C) - 1) * (m + 1), preprocessed_file);
    auto B2_mults = load_points_affine<ECpe>(((1U << C) - 1) * (m + 1), preprocessed_file);
    auto L_mults = load_points_affine<ECp>(((1U << C) - 1) * (m - 1), preprocessed_file);
    fclose(preprocessed_file);

    FILE *inputs_file = fopen(input_path, "r");
    auto w_ = load_scalars(m + 1, inputs_file);
    const var *w = w_.get(); // GPU使用的标量
    rewind(inputs_file);
    auto inputs = ppt::read_input(inputs_file, d, m); // CPU使用的标量
    fclose(inputs_file);
    //G1 zero = G1::G1_zero;

}*/