#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <assert.h>
#define NDEBUG 1

#include <prover_reference_functions.hpp>

#include "multiexp/reduce.cu"

class Semaphore
{
public:
    Semaphore(int count_ = 0)
        : count(count_)
    {
    }

    inline void notify()
    {
        std::unique_lock<std::mutex> lock(mtx);
        count++;
        // notify the waiting thread
        cv.notify_one();
    }
    inline void wait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        while (count == 0)
        {
            // wait on the mutex until notify is called
            cv.wait(lock);
        }
        count--;
    }

private:
    std::mutex mtx;
    std::condition_variable cv;
    int count;
};
// This is where all the FFTs happen

// template over the bundle of types and functions.
// Overwrites ca!
template <typename B>
typename B::vector_Fr *compute_H(size_t d, typename B::vector_Fr *ca,
                                 typename B::vector_Fr *cb,
                                 typename B::vector_Fr *cc)
{
    // TODO: 并行实现domain_iFFT,domain_cosetFFT,domain_cosetiFFT
    auto domain = B::get_evaluation_domain(d + 1);

    B::domain_iFFT(domain, ca);
    B::domain_iFFT(domain, cb);

    B::domain_cosetFFT(domain, ca);
    B::domain_cosetFFT(domain, cb);

    // Use ca to store H
    auto H_tmp = ca;

    size_t m = B::domain_get_m(domain);
    // for i in 0 to m: H_tmp[i] *= cb[i]
    B::vector_Fr_muleq(H_tmp, cb, m);

    B::domain_iFFT(domain, cc);
    B::domain_cosetFFT(domain, cc);

    m = B::domain_get_m(domain);

    // for i in 0 to m: H_tmp[i] -= cc[i]
    B::vector_Fr_subeq(H_tmp, cc, m);

    B::domain_divide_by_Z_on_coset(domain, H_tmp);

    B::domain_icosetFFT(domain, H_tmp);

    m = B::domain_get_m(domain);
    typename B::vector_Fr *H_res = B::vector_Fr_zeros(m + 1);
    B::vector_Fr_copy_into(H_tmp, H_res, m);
    return H_res;
}

template <typename B>
struct ec_type;

template <>
struct ec_type<mnt4753_libsnark>
{
    typedef ECp_MNT4 ECp;
    typedef ECp2_MNT4 ECpe;
};

template <>
struct ec_type<mnt6753_libsnark>
{
    typedef ECp_MNT6 ECp;
    typedef ECp3_MNT6 ECpe;
};
template <typename B, int C>
class ExponentVec
{
public:
    std::mutex mtx;
    typedef typename B::vector_Fr vector_Fr;
    vector_Fr *vec;
    var *norm_exp;

private:
    var *idxbuf;
    var *searr;

public:
    bool vecValid = false;
    bool vecOccupied = false;

    bool normValid = false;
    bool normAlloced = false;
    bool normOccupied = false;

    bool scanValid = false;
    bool scanAlloced = false;
    bool scanOccupied = false;

    bool globaled = false;

public:
    size_t N = 0;
    static constexpr unsigned MAX_WIN_IDX = (753 + C - 1) / C;
    static constexpr var WIN_SIZE = 1ULL << C;
    static constexpr size_t searr_size = (MAX_WIN_IDX * WIN_SIZE * 2) * sizeof(var);
    size_t idxbuf_size() { return N * MAX_WIN_IDX * sizeof(var); }
    ExponentVec(vector_Fr *vec, size_t N)
    {
        this->vec = vec;
        this->N = N;
        if (B::get_sizeFr(vec) != N)
        {
            printf("Warning: Vec size doesn't correspond to N! vecsize=%lu,N=%lu\n", B::get_sizeFr(vec), N);
        }
        vecValid = true;
        allocNormExpUnsafe();
        validNormUnsafe();
        allocScanUnsafe();
    }
    ExponentVec(var *norm_exp, size_t N)
    {
        this->norm_exp = norm_exp;
        this->N = N;
        normValid = true;
        normAlloced = true;
        allocScanUnsafe();
    }

    ExponentVec(size_t N, bool allocNorm = true, bool allocScan = true)
    {
        this->N = N;
        if (allocNorm)
            allocNormExpUnsafe();
        if (allocScan)
            allocScanUnsafe();
    }
    void allocNormExpUnsafe()
    {
        if (!normAlloced)
        {
            size_t vecN = B::get_sizeFr(vec);
            //validNormexp完全根据vec的长度进行计算
            //当vec与N不匹配时不得不申请额外的空间避免越界
            printf("Malloc for NormExp %lu Bytes\n", vecN * ELT_BYTES);
            norm_exp = (var *)malloc(vecN * ELT_BYTES);
            assert(norm_exp != nullptr);
            normAlloced = true;
            normOccupied = true;
        }
    }
    void allocScanUnsafe()
    {
        if (!scanAlloced)
        {
            cudaError_t e;
            printf("cudaMallocManaged %lu Bytes\n", idxbuf_size());
            e = cudaMallocManaged(&idxbuf, idxbuf_size(), cudaMemAttachHost);
            if (e != 0)
            {
                printf("cudaMallocManaged %d %s\n", e, cudaGetErrorString(e));
                abort();
            }
            printf("cudaMallocManaged %lu Bytes\n", searr_size);
            e = cudaMallocManaged(&searr, searr_size, cudaMemAttachHost);
            if (e != 0)
            {
                printf("cudaMallocManaged %d %s\n", e, cudaGetErrorString(e));
                abort();
            }
            scanAlloced = true;
            scanOccupied = true;
        }
    }
    void validNormUnsafe()
    {
        if (normValid)
            return;
        assert(vecValid);
        allocNormExpUnsafe();
        B::as_norm_arrayFr(norm_exp, vec);
        normValid = true;
        // printf("Valid Norm Success\n");
    }
    void validNorm()
    {
        mtx.lock();
        validNormUnsafe();
        mtx.unlock();
    }

    void validScanUnsafe()
    {
        if (scanValid)
            return;
        if (!normValid)
            validNormUnsafe();
        allocScanUnsafe();
        memset(norm_exp, N * ELT_BYTES, 0);
        prescan_better<C>(idxbuf, searr, norm_exp, N);
        // printf("Scan Success\n");
        scanValid = true;
    }

    void validScan()
    {
        mtx.lock();
        validScanUnsafe();
        mtx.unlock();
    }

    void deviceGlobal()
    {
        cudaStreamAttachMemAsync(NULL, idxbuf, idxbuf_size(), cudaMemAttachGlobal);
        cudaStreamAttachMemAsync(NULL, searr, idxbuf_size(), cudaMemAttachGlobal);
        cudaDeviceSynchronize();
        this->globaled = true;
    }

    void streamEnable(cudaStream_t &strm)
    {
        cudaStreamAttachMemAsync(strm, idxbuf, idxbuf_size());
        cudaStreamAttachMemAsync(strm, searr, idxbuf_size());
    }

    var *get_idxbuf()
    {
        return idxbuf;
    }
    var *get_searr()
    {
        return searr;
    }
    ~ExponentVec()
    {
        if (vecOccupied)
            B::delete_vector_Fr(vec);
        if (normOccupied)
            delete norm_exp;
        if (scanOccupied)
        {
            cudaFree(idxbuf);
            cudaFree(searr);
        }
    }
};
template <typename B, int C>
class gpu_funcs
{
public:
    typedef typename ec_type<B>::ECp ECp;
    typedef typename ec_type<B>::ECpe ECpe;
    typedef typename B::G1 G1;
    typedef typename B::G2 G2;
    typedef typename B::vector_G1 vector_G1;
    typedef typename B::vector_G2 vector_G2;
    typedef typename B::vector_Fr vector_Fr;

    static void multiexpG1_sync(G1 *&ret, ExponentVec<B, C> *expvec, vector_G1 *gvec,
                                size_t N, Semaphore *launched = nullptr)
    {
        //所有输入输出均在CPU内存
        auto begin = std::chrono::high_resolution_clock::now();
        auto t = begin;

        cudaStream_t strm;
        cudaStreamCreate(&strm);
        // my_print_time(t, "stream created");

        expvec->validScan();
        if (!expvec->globaled)
        {
            expvec->mtx.lock();
            expvec->streamEnable(strm);
        }
        var *idxbuf = expvec->get_idxbuf(), *searr = expvec->get_searr();
        // my_print_time(t, "exponent prepared");

        typedef ECp EC;
        static constexpr unsigned MAX_WIN_IDX = (753 + C - 1) / C;
        static constexpr var WIN_SIZE = 1ULL << C;

        var *mults, *outbuf;

        cudaMallocManaged(&mults, N * 2 * EC::field_type::DEGREE * ELT_BYTES, cudaMemAttachHost);
        cudaMallocManaged(&outbuf, MAX_WIN_IDX * WIN_SIZE * EC::NELTS * ELT_BYTES, cudaMemAttachHost);
        // my_print_time(t, "malloced");

        B::as_affine_arrayG1(mults, gvec);
        // my_print_time(t, "transformed");

        cudaStreamAttachMemAsync(strm, mults, N * 2 * EC::field_type::DEGREE * ELT_BYTES);
        cudaStreamAttachMemAsync(strm, outbuf, MAX_WIN_IDX * WIN_SIZE * EC::NELTS * ELT_BYTES);
        my_print_time(t, "Data prepared");

        ec_reduce_pippenger_unify<EC, C>(strm, outbuf, mults, idxbuf, searr, N);
        if (launched != nullptr)
            launched->notify();
        my_print_time(begin, "GPU Launched");
        cudaStreamSynchronize(strm);
        ret = B::read_pt_ECp(outbuf);
        cudaFree(mults);
        cudaFree(outbuf);
        cudaStreamDestroy(strm);
        if (!expvec->globaled)
        {
            expvec->mtx.unlock();
        }
        my_print_time(t, "Exit");
    }

    static void multiexpG2_sync(G2 *&ret, ExponentVec<B, C> *expvec, vector_G2 *gvec,
                                size_t N, Semaphore *launched = nullptr)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        auto t = begin;
        cudaStream_t strm;
        cudaStreamCreate(&strm);
        // my_print_time(t, "stream created");

        expvec->validScan();
        if (!expvec->globaled)
        {
            expvec->mtx.lock();
            expvec->streamEnable(strm);
        }
        var *idxbuf = expvec->get_idxbuf(), *searr = expvec->get_searr();
        // my_print_time(t, "exponent prepared");

        typedef ECpe EC;
        static constexpr unsigned MAX_WIN_IDX = (753 + C - 1) / C;
        static constexpr var WIN_SIZE = 1ULL << C;

        var *mults, *outbuf;

        cudaMallocManaged(&mults, N * 2 * EC::field_type::DEGREE * ELT_BYTES, cudaMemAttachHost);
        cudaMallocManaged(&outbuf, MAX_WIN_IDX * WIN_SIZE * EC::NELTS * ELT_BYTES, cudaMemAttachHost);
        // my_print_time(t, "malloced");

        B::as_affine_arrayG2(mults, gvec);
        // my_print_time(t, "transformed");

        cudaStreamAttachMemAsync(strm, mults, N * 2 * EC::field_type::DEGREE * ELT_BYTES);
        cudaStreamAttachMemAsync(strm, outbuf, MAX_WIN_IDX * WIN_SIZE * EC::NELTS * ELT_BYTES);
        my_print_time(t, "Data prepared");
        ec_reduce_pippenger_unify<EC, C>(strm, outbuf, mults, idxbuf, searr, N);
        if (launched != nullptr)
            launched->notify();
        my_print_time(begin, "GPU Launched");
        cudaStreamSynchronize(strm);
        ret = B::read_pt_ECpe(outbuf);
        cudaFree(mults);
        cudaFree(outbuf);
        cudaStreamDestroy(strm);
        if (!expvec->globaled)
        {
            expvec->mtx.unlock();
        }
        my_print_time(t, "Exit");
    }
};

void check_trailing(FILE *f, const char *name)
{
    long bytes_remaining = 0;
    while (fgetc(f) != EOF)
        ++bytes_remaining;
    if (bytes_remaining > 0)
        fprintf(stderr, "!! Trailing characters in \"%s\": %ld\n", name, bytes_remaining);
}

static inline auto now() -> decltype(std::chrono::high_resolution_clock::now())
{
    return std::chrono::high_resolution_clock::now();
}

template <typename T>
void print_time(T &t1, const char *str)
{
    auto t2 = std::chrono::high_resolution_clock::now();
    auto tim = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("%s: %ld ms\n", str, tim);
    t1 = t2;
}

template <typename B>
void run_prover(
    const char *params_path,
    const char *input_path,
    const char *output_path,
    const char *preprocessed_path)
{
    B::init_public_params();

    size_t primary_input_size = 1;

    auto beginning = now();
    auto t = beginning;

    FILE *params_file = fopen(params_path, "r");
    size_t d = read_size_t(params_file);
    size_t m = read_size_t(params_file);
    rewind(params_file);

    printf("d = %zu, m = %zu\n", d, m);

    typedef typename ec_type<B>::ECp ECp;
    typedef typename ec_type<B>::ECpe ECpe;

    typedef typename B::G1 G1;
    typedef typename B::G2 G2;

    static constexpr int R = 32;
    static constexpr int C = 5;
    FILE *preprocessed_file = fopen(preprocessed_path, "r");

    size_t space = ((m + 1) + R - 1) / R;

    // auto A_mults = load_points_affine<ECp>(((1U << C) - 1)*(m + 1), preprocessed_file);
    // auto out_A = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    auto B1_mults = load_points_affine<ECp>(((1U << C) - 1) * (m + 1), preprocessed_file);
    auto out_B1 = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    auto B2_mults = load_points_affine<ECpe>(((1U << C) - 1) * (m + 1), preprocessed_file);
    auto out_B2 = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    auto L_mults = load_points_affine<ECp>(((1U << C) - 1) * (m - 1), preprocessed_file);
    auto out_L = allocate_memory(space * ECpe::NELTS * ELT_BYTES);

    fclose(preprocessed_file);

    print_time(t, "load preprocessing");

    auto params = B::read_params(params_file, d, m);
    fclose(params_file);
    print_time(t, "load params");

    auto t_main = t;

    FILE *inputs_file = fopen(input_path, "r");
    auto w_ = load_scalars(m + 1, inputs_file);
    rewind(inputs_file);
    auto inputs = B::read_input(inputs_file, d, m);
    fclose(inputs_file);
    print_time(t, "load inputs");

    const var *w = w_.get();

    auto t_gpu = t;

    cudaStream_t sA, sB1, sB2, sL;

    // ec_reduce_straus<ECp, C, R>(sA, out_A.get(), A_mults.get(), w, m + 1);
    ec_reduce_straus<ECp, C, R>(sB1, out_B1.get(), B1_mults.get(), w, m + 1);
    ec_reduce_straus<ECpe, C, 2 * R>(sB2, out_B2.get(), B2_mults.get(), w, m + 1);
    ec_reduce_straus<ECp, C, R>(sL, out_L.get(), L_mults.get(), w + (primary_input_size + 1) * ELT_LIMBS, m - 1);

    //cudaDeviceSynchronize();
    //print_time(t, "Single Test G2");
    //exit(0);
    // ec_reduce_test<ECp, C, R>(sB1, out_B1.get(), B1_mults.get(), w, m + 1);
    // ec_reduce_test<ECpe, C, 2 * R>(sB2, out_B2.get(), B2_mults.get(), w, m + 1);
    // ec_reduce_test<ECp, C, R>(sL, out_L.get(), L_mults.get(), w + (primary_input_size + 1) * ELT_LIMBS, m - 1);
    print_time(t, "gpu launch");

    G1 *evaluation_At = B::multiexp_G1(B::input_w(inputs), B::params_A(params), m + 1);
    // G1 *evaluation_Bt1 = B::multiexp_G1(B::input_w(inputs), B::params_B1(params), m + 1);
    // G2 *evaluation_Bt2 = B::multiexp_G2(B::input_w(inputs), B::params_B2(params), m + 1);

    // Do calculations relating to H on CPU after having set the GPU in
    // motion
    auto H = B::params_H(params);
    auto coefficients_for_H =
        compute_H<B>(d, B::input_ca(inputs), B::input_cb(inputs), B::input_cc(inputs));
    G1 *evaluation_Ht = B::multiexp_G1(coefficients_for_H, H, d);

    print_time(t, "cpu 1");

    cudaDeviceSynchronize();
    // cudaStreamSynchronize(sA);
    // G1 *evaluation_At = B::read_pt_ECp(out_A.get());

    cudaStreamSynchronize(sB1);
    G1 *evaluation_Bt1 = B::read_pt_ECp(out_B1.get());

    cudaStreamSynchronize(sB2);
    G2 *evaluation_Bt2 = B::read_pt_ECpe(out_B2.get());

    cudaStreamSynchronize(sL);
    G1 *evaluation_Lt = B::read_pt_ECp(out_L.get());

    print_time(t_gpu, "gpu e2e");

    auto scaled_Bt1 = B::G1_scale(B::input_r(inputs), evaluation_Bt1);
    auto Lt1_plus_scaled_Bt1 = B::G1_add(evaluation_Lt, scaled_Bt1);
    auto final_C = B::G1_add(evaluation_Ht, Lt1_plus_scaled_Bt1);

    print_time(t, "cpu 2");

    B::groth16_output_write(evaluation_At, evaluation_Bt2, final_C, output_path);

    print_time(t, "store");

    print_time(t_main, "Total time from input to output: ");

    // cudaStreamDestroy(sA);
    cudaStreamDestroy(sB1);
    cudaStreamDestroy(sB2);
    cudaStreamDestroy(sL);

    B::delete_vector_G1(H);

    B::delete_G1(evaluation_At);
    B::delete_G1(evaluation_Bt1);
    B::delete_G2(evaluation_Bt2);
    B::delete_G1(evaluation_Ht);
    B::delete_G1(evaluation_Lt);
    B::delete_G1(scaled_Bt1);
    B::delete_G1(Lt1_plus_scaled_Bt1);
    B::delete_vector_Fr(coefficients_for_H);
    B::delete_groth16_input(inputs);
    B::delete_groth16_params(params);

    print_time(t, "cleanup");
    print_time(beginning, "Total runtime (incl. file reads)");
}

template <typename B, int C>
void run_prover_pippenger(
    const char *params_path,
    const char *input_path,
    const char *output_path)
{
    B::init_public_params();

    size_t primary_input_size = 1;

    auto beginning = now();
    auto t = beginning;

    FILE *params_file = fopen(params_path, "r");
    size_t d = read_size_t(params_file);
    size_t m = read_size_t(params_file);
    rewind(params_file);

    printf("d = %zu, m = %zu\n", d, m);

    typedef typename ec_type<B>::ECp ECp;
    typedef typename ec_type<B>::ECpe ECpe;

    typedef typename B::G1 G1;
    typedef typename B::G2 G2;

    // static constexpr int C = 7;

    print_time(t, "load preprocessing");

    auto params = B::read_params(params_file, d, m);
    fclose(params_file);
    print_time(t, "load params");

    auto t_main = t;

    FILE *inputs_file = fopen(input_path, "r");
    auto inputs = B::read_input(inputs_file, d, m);
    fclose(inputs_file);

    ExponentVec<B, C> expvecW(B::input_w(inputs), m + 1);
    expvecW.validNorm();
    expvecW.vecOccupied = true; // input_w向量最终由expvecW负责释放
    ExponentVec<B, C> expvecL(expvecW.norm_exp + (primary_input_size + 1) * ELT_LIMBS, m - 1);
    print_time(t, "load inputs");

    G1 *evaluation_Bt1;
    G2 *evaluation_Bt2;
    G1 *evaluation_Lt;
    G1 *evaluation_At;
    G1 *evaluation_Ht;
    auto t_gpu = t;
    Semaphore barrier(0);
    expvecW.validScan();
    expvecW.deviceGlobal();
    print_time(t, "sW Valid");
    expvecL.validScan();
    print_time(t, "sL Valid");

    std::thread tB2(gpu_funcs<B, C>::multiexpG2_sync, std::ref(evaluation_Bt2), &expvecW, B::params_B2(params), m + 1, &barrier);
    //tB2.join();
    //print_time(t, "Single Test G2");
    //exit(0);
    std::thread tB1(gpu_funcs<B, C>::multiexpG1_sync, std::ref(evaluation_Bt1), &expvecW, B::params_B1(params), m + 1, &barrier);
    std::thread tL(gpu_funcs<B, C>::multiexpG1_sync, std::ref(evaluation_Lt), &expvecL, B::params_L(params), m - 1, &barrier);
    std::thread tA(gpu_funcs<B, C>::multiexpG1_sync, std::ref(evaluation_At), &expvecW, B::params_A(params), m + 1, &barrier);

    barrier.wait();
    barrier.wait();
    barrier.wait();
    barrier.wait();

    // GPU Launch成功后才进行CPU计算，避免抢占CPU资源
    print_time(t, "gpu launch");
    // evaluation_At = B::multiexp_G1(B::input_w(inputs), B::params_A(params), m + 1);
    // evaluation_Bt1 = B::multiexp_G1(B::input_w(inputs), B::params_B1(params), m + 1);
    // evaluation_Bt2 = B::multiexp_G2(B::input_w(inputs), B::params_B2(params), m + 1);

    // Do calculations relating to H on CPU after having set the GPU in
    // motion
    auto H = B::params_H(params);
    auto coefficients_for_H =
        compute_H<B>(d, B::input_ca(inputs), B::input_cb(inputs), B::input_cc(inputs));
    ExponentVec<B, C> expvecH(coefficients_for_H, d);
    expvecH.validScan();
    std::thread tH(gpu_funcs<B, C>::multiexpG1_sync, std::ref(evaluation_Ht), &expvecH, H, d, &barrier);
    barrier.wait();
    // evaluation_Ht = B::multiexp_G1(coefficients_for_H, H, d);

    print_time(t, "cpu 1");

    tB1.join();
    tB2.join();
    tL.join();
    tA.join();
    tH.join();

    print_time(t_gpu, "gpu e2e");

    auto scaled_Bt1 = B::G1_scale(B::input_r(inputs), evaluation_Bt1);
    auto Lt1_plus_scaled_Bt1 = B::G1_add(evaluation_Lt, scaled_Bt1);
    auto final_C = B::G1_add(evaluation_Ht, Lt1_plus_scaled_Bt1);

    print_time(t, "cpu 2");

    B::groth16_output_write(evaluation_At, evaluation_Bt2, final_C, output_path);

    print_time(t, "store");

    print_time(t_main, "Total time from input to output: ");

    B::delete_vector_G1(H);

    B::delete_G1(evaluation_At);
    B::delete_G1(evaluation_Bt1);
    B::delete_G2(evaluation_Bt2);
    B::delete_G1(evaluation_Ht);
    B::delete_G1(evaluation_Lt);
    B::delete_G1(scaled_Bt1);
    B::delete_G1(Lt1_plus_scaled_Bt1);
    B::delete_vector_Fr(coefficients_for_H);
    B::delete_groth16_input(inputs);
    B::delete_groth16_params(params);

    print_time(t, "cleanup");
    print_time(beginning, "Total runtime (incl. file reads)");
}

int main(int argc, char **argv)
{
    setbuf(stdout, NULL);
    std::string curve(argv[1]);
    std::string mode(argv[2]);
    std::string method(argv[6]);

    const char *params_path = argv[3];

    if (mode == "compute")
    {
        const char *input_path = argv[4];
        const char *output_path = argv[5];
        if (method == "straus")
        {
            std::string preprocessed_path = argv[7];
            if (curve == "MNT4753")
                run_prover<mnt4753_libsnark>(params_path, input_path, output_path, preprocessed_path.c_str());
            else if (curve == "MNT6753")
                run_prover<mnt6753_libsnark>(params_path, input_path, output_path, preprocessed_path.c_str());
        }
        else if (method == "pippenger")
        {
            int C = atoi(argv[7]);
            if (curve == "MNT4753")
            {
                run_prover_pippenger<mnt4753_libsnark, 7>(params_path, input_path, output_path);
                /*
                                switch (C)
                                {
                                case 5:
                                    run_prover_pippenger<mnt4753_libsnark, 5>(params_path, input_path, output_path);
                                    break;
                                case 6:
                                    run_prover_pippenger<mnt4753_libsnark, 6>(params_path, input_path, output_path);
                                    break;
                                case 7:
                                    run_prover_pippenger<mnt4753_libsnark, 7>(params_path, input_path, output_path);
                                    break;
                                case 8:
                                    run_prover_pippenger<mnt4753_libsnark, 8>(params_path, input_path, output_path);
                                    break;
                                case 9:
                                    run_prover_pippenger<mnt4753_libsnark, 9>(params_path, input_path, output_path);
                                    break;
                                default:
                                    run_prover_pippenger<mnt4753_libsnark, 10>(params_path, input_path, output_path);
                                    break;
                                }
                                */
            }
            else if (curve == "MNT6753")
            {
                run_prover_pippenger<mnt6753_libsnark, 7>(params_path, input_path, output_path);
                /*
                                switch (C)
                                {
                                case 5:
                                    run_prover_pippenger<mnt6753_libsnark, 5>(params_path, input_path, output_path);
                                    break;
                                case 6:
                                    run_prover_pippenger<mnt6753_libsnark, 6>(params_path, input_path, output_path);
                                    break;
                                case 7:
                                    run_prover_pippenger<mnt6753_libsnark, 7>(params_path, input_path, output_path);
                                    break;
                                case 8:
                                    run_prover_pippenger<mnt6753_libsnark, 8>(params_path, input_path, output_path);
                                    break;
                                case 9:
                                    run_prover_pippenger<mnt6753_libsnark, 9>(params_path, input_path, output_path);
                                    break;
                                default:
                                    run_prover_pippenger<mnt6753_libsnark, 10>(params_path, input_path, output_path);
                                    break;
                                }
                                */
            }
        }
        else
            printf("Method Not Found!\n");
    }
    else if (mode == "preprocess")
    {
#if 0
      if (curve == "MNT4753") {
          run_preprocess<mnt4753_libsnark>(params_path);
      } else if (curve == "MNT6753") {
          run_preprocess<mnt4753_libsnark>(params_path);
      }
#endif
    }

    return 0;
}

/*
    print_time(t, "We are doing prescan test!");
    prescan<ECp, C>(cpu_B1_idxbuf, cpu_B1_searr, norm_exp, m + 1);
    print_time(t, "base prescan");
    prescan_better<ECp, C>(cpu_B2_idxbuf, cpu_B2_searr,norm_exp, m + 1);
    print_time(t, "better prescan");
    for (int i = 0; i < (m + 1) * MAX_WIN_IDX; i++)
    {
        if (cpu_B1_idxbuf[i] != cpu_B2_idxbuf[i])
        {
            printf("Idxbuf Check Failed! At %d needValue=%lx calValue=%lx\n",i,cpu_B1_idxbuf[i],cpu_B2_idxbuf[i]);
            dump_data("idxbuf", (m + 1) * MAX_WIN_IDX, cpu_B1_idxbuf);
            dump_data("idxbuf_cal", (m + 1) * MAX_WIN_IDX, cpu_B2_idxbuf);
            break;
        }
    }
    for (int i = 0; i < 2 * MAX_WIN_IDX * WIN_SIZE; i++)
    {
        if (cpu_B1_searr[i] != cpu_B2_searr[i])
        {
            printf("Searr Check Failed!\n");
            dump_data("searr", 2 * MAX_WIN_IDX * WIN_SIZE, cpu_B1_searr);
            dump_data("searr_cal", 2 * MAX_WIN_IDX * WIN_SIZE, cpu_B2_searr);
            break;
        }
    }
    exit(0);
*/