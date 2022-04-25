#pragma once
#include <cstdint>
#include <vector>
#include <chrono>
#include <memory>
#include <cooperative_groups.h>

#include "curves.cu"

static size_t read_size_t(FILE *input)
{
    size_t n;
    fread((void *)&n, sizeof(size_t), 1, input);
    return n;
}
static inline double as_mebibytes(size_t n)
{
    return n / (long double)(1UL << 20);
}

void print_meminfo(size_t allocated)
{
    size_t free_mem, dev_mem;
    cudaMemGetInfo(&free_mem, &dev_mem);
    fprintf(stderr, "Allocated %zu bytes; device has %.1f MiB free (%.1f%%).\n",
            allocated,
            as_mebibytes(free_mem),
            100.0 * free_mem / dev_mem);
}

struct CudaFree
{
    void operator()(var *mem) { cudaFree(mem); }
};
typedef std::unique_ptr<var, CudaFree> var_ptr;

var_ptr
allocate_memory(size_t nbytes, int dbg = 0)
{
    var *mem = nullptr;
    cudaMallocManaged(&mem, nbytes);
    if (mem == nullptr)
    {
        fprintf(stderr, "Failed to allocate enough device memory\n");
        abort();
    }
    if (dbg)
        print_meminfo(nbytes);
    return var_ptr(mem);
}

var_ptr
load_scalars(size_t n, FILE *inputs)
{
    static constexpr size_t scalar_bytes = ELT_BYTES;
    size_t total_bytes = n * scalar_bytes;

    auto mem = allocate_memory(total_bytes);
    if (fread((void *)mem.get(), total_bytes, 1, inputs) < 1)
    {
        fprintf(stderr, "Failed to read scalars\n");
        abort();
    }
    return mem;
}

template <typename EC>
var_ptr
load_points(size_t n, FILE *inputs)
{
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * ELT_BYTES;
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;
    static constexpr size_t jac_pt_bytes = 3 * coord_bytes;

    size_t total_aff_bytes = n * aff_pt_bytes;
    size_t total_jac_bytes = n * jac_pt_bytes;

    auto mem = allocate_memory(total_jac_bytes);
    if (fread((void *)mem.get(), total_aff_bytes, 1, inputs) < 1)
    {
        fprintf(stderr, "Failed to read all curve poinst\n");
        abort();
    }

    // insert space for z-coordinates
    char *cmem = reinterpret_cast<char *>(mem.get()); // lazy
    for (size_t i = n - 1; i > 0; --i)
    {
        char tmp_pt[aff_pt_bytes];
        memcpy(tmp_pt, cmem + i * aff_pt_bytes, aff_pt_bytes);
        memcpy(cmem + i * jac_pt_bytes, tmp_pt, aff_pt_bytes);
    }
    return mem;
}

template <typename EC>
var_ptr
load_points_affine(size_t n, FILE *inputs)
{
    typedef typename EC::field_type FF;

    static constexpr size_t coord_bytes = FF::DEGREE * ELT_BYTES;
    static constexpr size_t aff_pt_bytes = 2 * coord_bytes;

    size_t total_aff_bytes = n * aff_pt_bytes;

    auto mem = allocate_memory(total_aff_bytes);
    if (fread((void *)mem.get(), total_aff_bytes, 1, inputs) < 1)
    {
        fprintf(stderr, "Failed to read all curve poinst\n");
        abort();
    }
    return mem;
}

void dump_data(char *filename, size_t n, var *data)
{
    FILE *f = fopen(filename, "w");
    fwrite(data, n, sizeof(var), f);
    fclose(f);
}