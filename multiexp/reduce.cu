#include <cstdint>
#include <vector>
#include <chrono>
#include <memory>
#include <cooperative_groups.h>
#include <thread>
#include <mutex>
#include <prover_reference_functions.hpp>
#include "curves.cu"
#include "oprs.cu"

template <typename EC>
__global__ void
ec_multiexp_jac2jac(var *X, const var *W, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n)
    {
        typedef typename EC::group_type Fr;
        EC x;
        Fr w;
        int x_off = idx * EC::NELTS * ELT_LIMBS;
        int w_off = idx * ELT_LIMBS;

        EC::load_jac(x, X + x_off);
        Fr::load(w, W + w_off);

        // We're given W in Monty form for some reason, so undo that.
        // Fr::from_monty(w, w);
        EC::mul(x, w.a, x);
        EC::store_jac(X + x_off, x);
    }
}

template <typename EC>
__global__ void
ec_multiexp_aff2jac(var *out, var *X, const var *W, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n)
    {
        typedef typename EC::group_type Fr;
        EC x;
        Fr w;
        int x_off = idx * 2 * EC::field_type::DEGREE * ELT_LIMBS;
        int o_off = idx * 3 * EC::field_type::DEGREE * ELT_LIMBS;
        int w_off = idx * ELT_LIMBS;

        EC::load_affine(x, X + x_off);
        Fr::load(w, W + w_off);
        // printf("Thread:%d NMw=0x%llx\n", T, w.a);
        //  We're given W in Monty form for some reason, so undo that.
        // Fr::from_monty(w, w);
        EC::mul(x, w.a, x);
        // printf("Thread:%d w=0x%llx\n", T, w.a);
        EC::store_jac(out + o_off, x);
    }
}

// C is the size of the precomputation
// R is the number of points we're handling per thread
template <typename EC, int C = 4, int RR = 8>
__global__ void
ec_multiexp_straus(var *out, const var *multiples_, const var *scalars_, size_t N)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    size_t n = (N + RR - 1) / RR;
    if (idx < n)
    {
        // TODO: Treat remainder separately so R can remain a compile time constant
        size_t R = (idx < n - 1) ? RR : (N % RR);

        typedef typename EC::group_type Fr;
        static constexpr int JAC_POINT_LIMBS = 3 * EC::field_type::DEGREE * ELT_LIMBS;
        static constexpr int AFF_POINT_LIMBS = 2 * EC::field_type::DEGREE * ELT_LIMBS;
        int out_off = idx * JAC_POINT_LIMBS;
        int m_off = idx * RR * AFF_POINT_LIMBS;
        int s_off = idx * RR * ELT_LIMBS;

        Fr scalars[RR];
        for (int j = 0; j < R; ++j)
        {
            Fr::load(scalars[j], scalars_ + s_off + j * ELT_LIMBS);
            Fr::from_monty(scalars[j], scalars[j]);
        }

        const var *multiples = multiples_ + m_off;
        // TODO: Consider loading multiples and/or scalars into shared memory

        // i is smallest multiple of C such that i > 753
        int i = C * ((753 + C - 1) / C); // C * ceiling(753/C)
        assert((i - C * 753) < C);
        static constexpr var C_MASK = (1U << C) - 1U;

        EC x;
        EC::set_zero(x);
        while (i >= C)
        {
            EC::mul_2exp<C>(x, x);
            i -= C;

            int q = i / digit::BITS, r = i % digit::BITS;
            for (int j = 0; j < R; ++j)
            {
                //(scalars[j][q] >> r) & C_MASK
                auto g = fixnum::layout();
                var s = g.shfl(scalars[j].a, q);
                var win = (s >> r) & C_MASK;
                // Handle case where C doesn't divide digit::BITS
                int bottom_bits = digit::BITS - r;
                // detect when window overlaps digit boundary
                if (bottom_bits < C)
                {
                    s = g.shfl(scalars[j].a, q + 1);
                    win |= (s << bottom_bits) & C_MASK;
                }
                if (win > 0)
                {
                    EC m;
                    // EC::add(x, x, multiples[win - 1][j]);
                    EC::load_affine(m, multiples + ((win - 1) * N + j) * AFF_POINT_LIMBS);
                    EC::mixed_add(x, x, m);
                }
            }
        }
        EC::store_jac(out + out_off, x);
    }
}

template <typename EC>
__global__ void
ec_multiexp(var *X, const var *W, size_t n)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    if (idx < n)
    {
        typedef typename EC::group_type Fr;
        EC x;
        Fr w;
        int x_off = idx * EC::NELTS * ELT_LIMBS;
        int w_off = idx * ELT_LIMBS;

        EC::load_affine(x, X + x_off);
        Fr::load(w, W + w_off);

        // We're given W in Monty form for some reason, so undo that.
        Fr::from_monty(w, w);
        EC::mul(x, w.a, x);

        EC::store_jac(X + x_off, x);
    }
}

static constexpr size_t threads_per_block = 256;

template <typename EC, int C, int R>
void ec_reduce_straus(cudaStream_t &strm, var *out, const var *multiples, const var *scalars, size_t N)
{
    cudaStreamCreate(&strm);

    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;
    size_t n = (N + R - 1) / R;

    size_t nblocks = (n * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    ec_multiexp_straus<EC, C, R><<<nblocks, threads_per_block, 0, strm>>>(out, multiples, scalars, N);

    size_t r = n & 1, m = n / 2;
    for (; m != 0; r = m & 1, m >>= 1)
    {
        nblocks = (m * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

        ec_sum_all_kernel<EC><<<nblocks, threads_per_block, 0, strm>>>(out, out + m * pt_limbs, m);
        if (r)
            ec_sum_all_kernel<EC><<<1, threads_per_block, 0, strm>>>(out, out + 2 * m * pt_limbs, 1);
    }
}

template <typename EC>
void ec_reduce(cudaStream_t &strm, var *X, const var *w, size_t n)
{
    cudaStreamCreate(&strm);

    size_t nblocks = (n * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    // FIXME: Only works on Pascal and later.
    // auto grid = cg::this_grid();
    ec_multiexp<EC><<<nblocks, threads_per_block, 0, strm>>>(X, w, n);

    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;

    size_t r = n & 1, m = n / 2;
    for (; m != 0; r = m & 1, m >>= 1)
    {
        nblocks = (m * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

        ec_sum_all_kernel<EC><<<nblocks, threads_per_block, 0, strm>>>(X, X + m * pt_limbs, m);
        if (r)
            ec_sum_all_kernel<EC><<<1, threads_per_block, 0, strm>>>(X, X + 2 * m * pt_limbs, 1);
        // TODO: Not sure this is really necessary.
        // grid.sync();
    }
}

template <typename EC>
void ec_reduce_jac2jac(cudaStream_t &strm, var *X, const var *w, size_t n)
{
    size_t nblocks = (n * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    // FIXME: Only works on Pascal and later.
    // auto grid = cg::this_grid();
    ec_multiexp_jac2jac<EC><<<nblocks, threads_per_block, 0, strm>>>(X, w, n);

    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;

    size_t r = n & 1, m = n / 2;
    for (; m != 0; r = m & 1, m >>= 1)
    {
        nblocks = (m * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

        ec_sum_all_kernel<EC><<<nblocks, threads_per_block, 0, strm>>>(X, X + m * pt_limbs, m);
        if (r)
            ec_sum_all_kernel<EC><<<1, threads_per_block, 0, strm>>>(X, X + 2 * m * pt_limbs, 1);
        // TODO: Not sure this is really necessary.
        // grid.sync();
    }
}

template <typename EC>
void ec_reduce_aff2jac(cudaStream_t &strm, var *out, var *X, const var *w, size_t n)
{
    size_t nblocks = (n * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    // FIXME: Only works on Pascal and later.
    // auto grid = cg::this_grid();
    ec_multiexp_aff2jac<EC><<<nblocks, threads_per_block, 0, strm>>>(out, X, w, n);

    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;

    size_t r = n & 1, m = n / 2;
    for (; m != 0; r = m & 1, m >>= 1)
    {
        nblocks = (m * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

        ec_sum_all_kernel<EC><<<nblocks, threads_per_block, 0, strm>>>(out, out + m * pt_limbs, m);
        if (r)
            ec_sum_all_kernel<EC><<<1, threads_per_block, 0, strm>>>(out, out + 2 * m * pt_limbs, 1);
        // TODO: Not sure this is really necessary.
        // grid.sync();
    }
}

static auto my_tick1 = std::chrono::high_resolution_clock::now();
template <typename T>
void my_print_time(T &t1, const char *str)
{
    auto t2 = std::chrono::high_resolution_clock::now();
    auto tim = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto tid = std::this_thread::get_id();
    printf("Thread%u\t%s: %ld ms\n", (*(uint32_t *)&tid), str, tim);
    t1 = t2;
}

template <typename EC, int R>
void ec_reduce_puresum(cudaStream_t &strm, var *out, var *jac_mult, const var *norm_exp, size_t N)
{
    cudaError_t e;
    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;
    auto tmp = allocate_memory(N * EC::NELTS * ELT_BYTES);
    var *buf = tmp.get();                //当前需要加和的数
    ec_setZ_jac_async<EC>(strm, out, 1); //初始化输出为EC::ZERO
    e = cudaStreamSynchronize(strm);
    if (e != 0)
        printf("Sync %d %s\n", e, cudaGetErrorString(e));
    for (int i = 752; i >= 0; i--)
    {
        ec_mul_2exp_jac_async<EC, 1>(strm, out, out, 1); //加倍
        e = cudaStreamSynchronize(strm);
        if (e != 0)
            printf("Sync %d %s\n", e, cudaGetErrorString(e));
        //同步，使CPU可以使用buf
        var buf_p = 0;
        int q = i / digit::BITS, r = i % digit::BITS;
        for (int j = 0; j < N; j++)
        {
            var win = (norm_exp[j * ELT_LIMBS + q] >> r) & 1ULL;
            if (win)
            {
                memcpy(buf + buf_p, jac_mult + j * pt_limbs, EC::NELTS * ELT_BYTES);
                buf_p += pt_limbs;
                // bucket[i] = VAR::add(bucket[i], base[j]);
            }
        }
        if (buf_p != 0)
        {
            ec_sum_reduce<EC>(strm, buf, buf_p / pt_limbs);
            ec_sum_all_jac_async<EC>(strm, out, buf, 1);
        }
    }
    e = cudaStreamSynchronize(strm);
    if (e != 0)
        printf("Sync %d %s\n", e, cudaGetErrorString(e));
}

template <typename EC, int R>
void ec_reduce_puresum_para(cudaStream_t &strm, var *out, var *jac_mult, const var *norm_exp, size_t N)
{
    cudaError_t e;
    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;
    auto tmp = allocate_memory(753 * N * EC::NELTS * ELT_BYTES);
    var *global_buf = tmp.get(); //当前需要加和的数
    var buf_ps[753];
    cudaStream_t streams[753];
    for (int i = 0; i < 753; i++)
    {
        cudaStreamCreate(streams + i);
    }
    for (int i = 752; i >= 0; i--)
    {
        var *buf = global_buf + i * N * EC::NELTS * ELT_BYTES;
        var buf_p = 0;
        int q = i / digit::BITS, r = i % digit::BITS;
        for (int j = 0; j < N; j++)
        {
            var win = (norm_exp[j * ELT_LIMBS + q] >> r) & 1ULL;
            if (win)
            {
                memcpy(buf + buf_p, jac_mult + j * pt_limbs, EC::NELTS * ELT_BYTES);
                buf_p += pt_limbs;
                // bucket[i] = VAR::add(bucket[i], base[j]);
            }
        }
        buf_ps[i] = buf_p;
    }
    for (int i = 752; i >= 0; i--)
    {
        var *buf = global_buf + i * N * EC::NELTS * ELT_BYTES;
        if (buf_ps[i] != 0)
        {
            ec_sum_reduce<EC>(streams[i], buf, buf_ps[i] / pt_limbs);
            ec_sum_all_jac_async<EC>(streams[i], out + i * pt_limbs, buf, 1);
        }
    }
    e = cudaDeviceSynchronize();
    if (e != 0)
        printf("Sync %d %s\n", e, cudaGetErrorString(e));
    for (int i = 0; i < 753; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    ec_setZ_jac_async<EC>(strm, global_buf, 1); //初始化输出为EC::ZERO
    for (int i = 752; i >= 0; i--)
    {
        ec_mul_2exp_jac_async<EC, 1>(strm, global_buf, global_buf, 1); //加倍
        if (buf_ps[i] != 0)
        {
            ec_sum_all_jac_async<EC>(strm, global_buf, out + i * pt_limbs, 1);
        }
    }
    ec_sum_all_jac_async<EC>(strm, out, global_buf, 1);
    e = cudaStreamSynchronize(strm);
    if (e != 0)
        printf("Sync %d %s\n", e, cudaGetErrorString(e));
}
template <typename EC, int C> //假设C<64
__global__ void ec_multiexp_pippenger_unify_kernel(var *out, const var *aff_mult, const var *idxbuf, const var *searr)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;
    int idx = elts_per_block * B + tileIdx;
    static constexpr int MAX_WIN_IDX = (753 + C - 1) / C;
    //    static constexpr var WIN_SIZE = 1ULL << C;
    //    static constexpr var C_MASK = (1ULL << C) - 1ULL;
    int win_idx = idx >> C;
    // var cwin = ((var)idx) & C_MASK;
    var o_off = idx * EC::NELTS * ELT_LIMBS;
    if (win_idx < MAX_WIN_IDX)
    {
        const var *idx_start = idxbuf + searr[idx * 2];
        const var *idx_end = idxbuf + searr[idx * 2 + 1];
        EC x;
        EC::set_zero(x);
        for (; idx_start != idx_end; idx_start++)
        {
            EC m;
            EC::load_affine(m, aff_mult + (*idx_start) * 2 * EC::field_type::DEGREE * ELT_LIMBS);
            EC::mixed_add(x, x, m);
        }
        EC::store_jac(out + o_off, x);
    }
}
template <typename EC, int C>
__global__ void ec_highestbit_reduce(var *jac_mult)
{
    const int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    const int elts_per_block = D / BIG_WIDTH;
    const int tileIdx = T / BIG_WIDTH;
    const int idx = elts_per_block * B + tileIdx;
    // const var ridx = idx + 1;

    static constexpr int MAX_WIN_IDX = (753 + C - 1) / C;
    static constexpr var WIN_SIZE = 1ULL << C;
//    static constexpr var C_MASK = (1ULL << C) - 1ULL;

    static constexpr int VALID_BIT = 753 % C;
    static constexpr var VALID_WIN_SIZE = 1ULL << VALID_BIT;
    // static constexpr var VALID_MASK = (1ULL << VALID_BIT) - 1ULL;
    if (idx < VALID_WIN_SIZE)
    {
        var *m_start = jac_mult + (MAX_WIN_IDX - 1) * WIN_SIZE * EC::NELTS * ELT_LIMBS;
        EC x;
        EC::set_zero(x);
        EC zero;
        EC::set_zero(zero);
        for (var cwin = idx; cwin < WIN_SIZE; cwin += VALID_WIN_SIZE)
        {
            EC m;
            EC::load_jac(m, m_start + cwin * EC::NELTS * ELT_LIMBS);
            EC::add(x, x, m);
            EC::store_jac(m_start + cwin * EC::NELTS * ELT_LIMBS, zero);
        }
        EC::store_jac(m_start + idx * EC::NELTS * ELT_LIMBS, x);
    }
}
template <typename EC, int C> //假设C<64
__global__ void ec_CMASKReduce_kernel(var *jac_mult)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;
    int win_idx = idx;
    static constexpr int MAX_WIN_IDX = (753 + C - 1) / C;
    static constexpr var WIN_SIZE = 1ULL << C;
    static constexpr var C_MASK = (1ULL << C) - 1ULL;
    if (win_idx < MAX_WIN_IDX)
    {
        var *m_start = jac_mult + win_idx * WIN_SIZE * EC::NELTS * ELT_LIMBS;
        EC x;
        EC::set_zero(x);
        EC running_sum;
        EC::set_zero(running_sum);
        for (int cwin = C_MASK; cwin > 0; cwin--)
        {
            EC m;
            EC::load_jac(m, m_start + cwin * EC::NELTS * ELT_LIMBS);
            EC::add(running_sum, running_sum, m);
            EC::add(x, x, running_sum);
        }
        EC::store_jac(m_start, x);
    }
}
template <typename EC, int C> //这个函数实质上是串行的
__global__ void ec_final_reduce_kernel(var *jac_mult)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;
    static constexpr int MAX_WIN_IDX = (753 + C - 1) / C;
    static constexpr var WIN_SIZE = 1ULL << C;
    if (idx == 0)
    {
        EC x;
        EC::set_zero(x);
        for (int win_idx = MAX_WIN_IDX - 1; win_idx >= 0; win_idx--)
        {
            if (!EC::is_zero(x))
            {
                EC::mul_2exp<C>(x, x);
            }
            var *m_start = jac_mult + win_idx * WIN_SIZE * EC::NELTS * ELT_LIMBS;
            EC m;
            EC::load_jac(m, m_start);
            EC::add(x, x, m);
        }
        EC::store_jac(jac_mult, x);
    }
}
static int count = 0;
template <int C>
void prescan_perwin(var *idxbuf, var *searr_, const var *norm_exp, size_t N, int win_idx)
{
    static constexpr var WIN_SIZE = 1ULL << C;
    static constexpr var C_MASK = WIN_SIZE - 1;
    // idxbuf的offset在count_sum上完成了
    var *searr = searr_ + win_idx * WIN_SIZE * 2; // offset，前面的每个window占用2*WIN_SIZE个searr
    int q = (win_idx * C) / digit::BITS, r = (win_idx * C) % digit::BITS;
    // Handle case where C doesn't divide digit::BITS
    int bottom_bits = digit::BITS - r;
    size_t cwin_count[WIN_SIZE];
    memset(cwin_count, 0, sizeof(size_t) * WIN_SIZE);
    for (int i = 0; i < N; i++)
    {
        //扫描一遍所有数据，统计当前window下各cwin有多少个值
        var s = norm_exp[i * ELT_LIMBS + q];
        var win = (s >> r) & C_MASK;
        // detect when window overlaps digit boundary
        if (bottom_bits < C)
        {
            s = norm_exp[i * ELT_LIMBS + q + 1];
            win |= (s << bottom_bits) & C_MASK;
        }
        cwin_count[win]++;
    }
    size_t count_sum = win_idx * N; // offset，前面的每个window占用N个idxbuf
    for (var cwin = 0; cwin < WIN_SIZE; cwin++)
    {
        searr[cwin * 2] = count_sum;
        count_sum += cwin_count[cwin];
        searr[cwin * 2 + 1] = count_sum;
    }
    memset(cwin_count, 0, sizeof(size_t) * WIN_SIZE); //重置以便再次扫描
    for (int i = 0; i < N; i++)
    {
        //再扫描一遍所有数据，将所有idx放入对应位置
        var s = norm_exp[i * ELT_LIMBS + q];
        var win = (s >> r) & C_MASK;
        // detect when window overlaps digit boundary
        if (bottom_bits < C)
        {
            s = norm_exp[i * ELT_LIMBS + q + 1];
            win |= (s << bottom_bits) & C_MASK;
        }
        idxbuf[searr[2 * win] + cwin_count[win]] = i; // searr[2*win]为目标区域起点，cwin_count统计了该区域已放入多少数据
        cwin_count[win]++;
    }
    static constexpr unsigned MAX_WIN_IDX = (753 + C - 1) / C;
    
    if (win_idx == MAX_WIN_IDX - 1)
    {
        static constexpr int VALID_BIT = 753 % C;
        static constexpr var VALID_WIN_SZIE = 1ULL << VALID_BIT;
        // static constexpr var VALID_MASK = (1ULL << VALID_BIT) - 1ULL;
        static constexpr size_t REP_TIMES = WIN_SIZE / VALID_WIN_SZIE;
        for (var cwin = 0; cwin < VALID_WIN_SZIE; cwin++)
        {
            //printf("cwin=%lx [%lu,%lu]\n", cwin, searr[2 * cwin], searr[2 * cwin + 1]);
            size_t total_point = searr[2 * cwin + 1] - searr[2 * cwin];
            const size_t max_point = (total_point + REP_TIMES - 1) / REP_TIMES;
            size_t rep_win = cwin;
            size_t cindex = searr[2 * cwin];
            while (true)
            {
                searr[rep_win * 2] = cindex;
                if (total_point < max_point || rep_win > C_MASK)
                {
                    searr[rep_win * 2 + 1] = cindex + total_point;
                    break;
                }
                cindex += max_point;
                searr[rep_win * 2 + 1] = cindex;
                total_point -= max_point;
                rep_win += VALID_WIN_SZIE;
            }
        }
    }
}
#ifndef MULTICORE
#define MULTICORE
#endif
//目前强制使用多核模式
template <int C>
void prescan_better(var *idxbuf, var *searr, const var *norm_exp, size_t N)
{
    static constexpr unsigned MAX_WIN_IDX = (753 + C - 1) / C;
#ifdef MULTICORE
    std::thread *trds[MAX_WIN_IDX];
    for (int win_idx = 1; win_idx < MAX_WIN_IDX; win_idx++)
    {
        trds[win_idx] = new std::thread(prescan_perwin<C>, idxbuf, searr, norm_exp, N, win_idx);
    }
    prescan_perwin<C>(idxbuf, searr, norm_exp, N, 0);
    for (int win_idx = 1; win_idx < MAX_WIN_IDX; win_idx++)
    {
        trds[win_idx]->join();
        delete trds[win_idx];
    }
#else
    for (int win_idx = 0; win_idx < MAX_WIN_IDX; win_idx++)
    {
        prescan_perwin<EC, C>(idxbuf, searr, norm_exp, N, win_idx);
    }
#endif
}

template <typename EC, int C>
void prescan(var *idxbuf, var *searr, const var *norm_exp, size_t N)
{
    /*
        duplicated
        初版方案使用最暴力方式执行prescan，经常导致CPU成为性能瓶颈
    */
    static constexpr unsigned MAX_WIN_IDX = (753 + C - 1) / C;
    static constexpr var WIN_SIZE = 1ULL << C;
    static constexpr var C_MASK = WIN_SIZE - 1;
    static constexpr unsigned TPB = 256;
    var idxbuf_idx = 0;
    var searr_idx = 0;
    for (int win_idx = 0; win_idx < MAX_WIN_IDX; win_idx++)
    {
        int q = (win_idx * C) / digit::BITS, r = (win_idx * C) % digit::BITS;
        // Handle case where C doesn't divide digit::BITS
        int bottom_bits = digit::BITS - r;
        for (var cwin = 0; cwin < WIN_SIZE; cwin++)
        {
            searr[searr_idx] = idxbuf_idx;
            for (int i = 0; i < N; i++)
            {
                var s = norm_exp[i * ELT_LIMBS + q];
                var win = (s >> r) & C_MASK;
                // detect when window overlaps digit boundary
                if (bottom_bits < C)
                {
                    s = norm_exp[i * ELT_LIMBS + q + 1];
                    win |= (s << bottom_bits) & C_MASK;
                }
                if (win == cwin)
                {
                    idxbuf[idxbuf_idx] = i;
                    idxbuf_idx++;
                }
            }
            searr_idx++;
            searr[searr_idx] = idxbuf_idx;
            searr_idx++;
        }
    }
    assert(idxbuf_idx == (N * MAX_WIN_IDX));
    assert(searr_idx == (2 * MAX_WIN_IDX * WIN_SIZE));
    /*
    count++;
    std::string filename = "show";
    filename += ('0' + count);
    filename += ".txt";
    FILE *f = fopen(filename.c_str(), "w");
    for (int i = 0; i < MAX_WIN_IDX; i++)
    {
        fprintf(f, "WIN_IDX=%d\t", i);
        for (var cwin = 0; cwin < WIN_SIZE; cwin++)
        {
            fprintf(f, " [%lu,%lu]", searr[i * WIN_SIZE * 2 + cwin * 2], searr[i * WIN_SIZE * 2 + cwin * 2 + 1]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    */
}

template <typename EC, int C> //假设752%C==0且C<64
void ec_reduce_pippenger_unify(cudaStream_t &strm, var *out, const var *aff_mult, const var *idxbuf, const var *searr, size_t N)
{
    /*
        var *test_mem;
        cudaMallocManaged(&test_mem, EC::NELTS * ELT_BYTES);*/
    //    cudaError_t e;
    static constexpr unsigned MAX_WIN_IDX = (753 + C - 1) / C;
    static constexpr var WIN_SIZE = 1ULL << C;
    // static constexpr var C_MASK = WIN_SIZE - 1;
    var tN = MAX_WIN_IDX * WIN_SIZE * BIG_WIDTH;
    static constexpr unsigned TPB = 256;
    var bN = (tN + TPB - 1) / TPB;
    // printf("Allocate 0x%lxBytes Memory\n", MAX_WIN_IDX * WIN_SIZE * EC::NELTS * ELT_BYTES);
    //  auto idxbuf_ptr = allocate_memory(N * MAX_WIN_IDX * sizeof(var));
    //  auto searr_ptr = allocate_memory((MAX_WIN_IDX * WIN_SIZE * 2) * sizeof(var));
    //  var *idxbuf = idxbuf_ptr.get();
    //  var *searr = searr_ptr.get();
    //  prescan<EC, C>(idxbuf, searr, norm_exp, N);
    printf("Unify kernel with bN=%lu,tN=0x%lx\n", bN, tN);
    ec_multiexp_pippenger_unify_kernel<EC, C><<<bN, TPB, 0, strm>>>(out, aff_mult, idxbuf, searr);
    static constexpr int VALID_BIT = 753 % C;
    static constexpr var VALID_MASK = (1ULL << VALID_BIT) - 1ULL;
    tN = (VALID_MASK + 1) * BIG_WIDTH;
    bN = (tN + TPB - 1) / TPB;
    ec_highestbit_reduce<EC, C><<<bN, TPB, 0, strm>>>(out);
    tN = MAX_WIN_IDX * BIG_WIDTH;
    bN = (tN + TPB - 1) / TPB;
    /*
        {
            // Test for Unify
            e = cudaStreamSynchronize(strm);
            if (e != 0)
                printf("Sync %d %s\n", e, cudaGetErrorString(e));
            int win_idx = 0;
            var cwin = 1UL;
            int win_start = win_idx * C;
            int q = win_start / digit::BITS, r = win_start % digit::BITS;
            ec_setZ_jac_async<EC>(strm, test_mem, 1);
            e = cudaStreamSynchronize(strm);
            for (int i = 0; i < N; i++)
            {
                var win = (norm_exp[i * ELT_LIMBS + q] >> r) & C_MASK;
                if (win == cwin)
                {
                    ec_sum_all_jac_async<EC>(strm, test_mem, jac_mult + i * EC::NELTS * ELT_LIMBS, 1);
                    fprintf(f, "%d ", i);
                }
                e = cudaStreamSynchronize(strm);
            }
            e = cudaStreamSynchronize(strm);
            printf("Unify Test:   %lx %lx\n", test_mem[0], out[(win_idx * (1 << C) + cwin) * EC::NELTS * ELT_LIMBS]);
            fclose(f);
        }
    */
    // printf("CMASK reduce with bN=%lu,tN=0x%lx\n", bN, tN);
    ec_CMASKReduce_kernel<EC, C><<<bN, TPB, 0, strm>>>(out);
    /*
    e = cudaStreamSynchronize(strm);
    if (e != 0)
        printf("Sync %d %s\n", e, cudaGetErrorString(e));
    */
    // printf("Final reduce with bN=1,tN=32\n");
    ec_final_reduce_kernel<EC, C><<<1, 32, 0, strm>>>(out);
    /*
    e = cudaStreamSynchronize(strm);
    if (e != 0)
        printf("Sync %d %s\n", e, cudaGetErrorString(e));
        */
}
template <typename EC, int C, int R>
void ec_reduce_test2(cudaStream_t &strm, var *out, var *multiples, const var *scalars, size_t N)
{
    auto tmp = allocate_memory(N * EC::NELTS * ELT_BYTES);
    cudaStreamCreate(&strm);
    ec_reduce_aff2jac<EC>(strm, tmp.get(), multiples, scalars, N);
    ec_cpy_jac2jac_async<EC>(strm, out, tmp.get(), 1);
}
