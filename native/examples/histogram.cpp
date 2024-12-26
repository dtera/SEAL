//
// Created by zhangfan on 2024/9/25.
//
// #include <immintrin.h>

#include "seal/seal.h"
#include "seal/util/clipnormal.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"

using namespace seal;
using namespace std;

Plaintext encode_non_ntt(SEALContext &context, const vector<double> &pod, double scale)
{
    auto &context_data = *context.first_context_data();
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    const auto &ntt_tables = context_data.small_ntt_tables();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

    if (pod.size() > coeff_count)
    {
        throw invalid_argument("encode_non_ntt: input size incorrect");
    }

    if (coeff_modulus_size != 1)
    {
        throw logic_error("encode2vec: only support with coeff_modulus_size = 1");
    }

    vector<double> values(coeff_count, 0.0);
    std::transform(pod.begin(), pod.end(), values.begin(), [&](double v) { return v * scale; });

    double max_coeff = 0;
    for (const auto &vv : values)
    {
        max_coeff = std::max(max_coeff, std::fabs(vv));
    }

    int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max<>(max_coeff, 1.0)))) + 1;
    if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
    {
        throw std::invalid_argument("encoded values are too large");
    }

    Plaintext plaintext;
    plaintext.parms_id() = parms_id_zero;
    plaintext.resize(util::mul_safe(coeff_count, coeff_modulus_size));

    for (size_t i = 0; i < coeff_count; i++)
    {
        double coeffd = std::round(values[i]);
        bool is_negative = std::signbit(coeffd);

        auto coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));

        if (is_negative)
        {
            plaintext[i] = util::negate_uint_mod(util::barrett_reduce_64(coeffu, coeff_modulus[0]), coeff_modulus[0]);
        }
        else
        {
            plaintext[i] = util::barrett_reduce_64(coeffu, coeff_modulus[0]);
        }
    }

    util::ntt_negacyclic_harvey(plaintext.data(), ntt_tables[0]);
    plaintext.parms_id() = context.first_parms_id();
    plaintext.scale() = scale;

    return plaintext;
}

Plaintext encode_non_ntt(SEALContext &context, const vector<int> &pod)
{
    auto &context_data = *context.first_context_data();
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    const auto &ntt_tables = context_data.small_ntt_tables();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

    if (pod.size() > coeff_count)
    {
        throw invalid_argument("encode_non_ntt: input size incorrect");
    }

    if (coeff_modulus_size != 1)
    {
        throw logic_error("encode2vec: only support with coeff_modulus_size = 1");
    }

    vector<int64_t> values(coeff_count, 0);
    for (size_t i = 0; i < pod.size(); i++)
    {
        values[i] = pod[i];
    }

    double max_coeff = 0;
    for (const auto &vv : values)
    {
        max_coeff = std::max(max_coeff, std::fabs(vv));
    }

    int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max<>(max_coeff, 1.0)))) + 1;
    if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
    {
        throw std::invalid_argument("encoded values are too large");
    }

    Plaintext plaintext;
    plaintext.parms_id() = parms_id_zero;
    plaintext.resize(util::mul_safe(coeff_count, coeff_modulus_size));

    for (std::size_t i = 0; i < coeff_count; i++)
    {
        int64_t coeffd = values[i];
        bool is_negative = std::signbit(coeffd);

        auto coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));

        if (is_negative)
        {
            plaintext[i] = util::negate_uint_mod(util::barrett_reduce_64(coeffu, coeff_modulus[0]), coeff_modulus[0]);
        }
        else
        {
            plaintext[i] = util::barrett_reduce_64(coeffu, coeff_modulus[0]);
        }
    }

    if (!plaintext.is_zero())
    {
        util::ntt_negacyclic_harvey(plaintext.data(), ntt_tables[0]);
    }

    plaintext.parms_id() = context.first_parms_id();
    plaintext.scale() = 1.0;

    return plaintext;
}

vector<double> decode_non_ntt(SEALContext &context, const Plaintext &plain)
{
    auto &context_data = *context.get_context_data(plain.parms_id());
    auto &parms = context_data.parms();
    std::size_t coeff_count = parms.poly_modulus_degree();
    auto ntt_tables = context_data.small_ntt_tables();

    vector<uint64_t> values(coeff_count);
    std::copy_n(plain.data(), coeff_count, values.data());

    util::inverse_ntt_negacyclic_harvey(values.data(), ntt_tables[0]);

    uint64_t mod = parms.coeff_modulus().front().value();

    vector<double> res(coeff_count);
    for (std::size_t i = 0; i < coeff_count; i++)
    {
        int64_t vv;
        if (values[i] * 2 > mod)
        {
            vv = static_cast<int64_t>(values[i] - mod);
        }
        else
        {
            vv = static_cast<int64_t>(values[i]);
        }
        res[i] = static_cast<double>(vv) / plain.scale();
    }

    return res;
}

vector<Plaintext> encode2vec(SEALContext &context, const vector<double> &arr1, const vector<double> &arr2, double scale)
{
    if (arr1.size() != arr2.size())
    {
        throw invalid_argument("encode2vec: input size incorrect");
    }

    auto &context_data = *context.first_context_data();
    auto &parms = context_data.parms();

    size_t coeff_count = parms.poly_modulus_degree();
    size_t slot_count = coeff_count / 2;

    vector<Plaintext> plain_vec;
    for (size_t i = 0; i < arr1.size(); i += slot_count)
    {
        size_t bgn = i;
        size_t end = std::min(arr1.size(), bgn + slot_count);

        vector<double> values(coeff_count);
        std::copy_n(arr1.data() + bgn, (end - bgn), values.data());
        std::copy_n(arr2.data() + bgn, (end - bgn), values.data() + slot_count);

        Plaintext plain = encode_non_ntt(context, values, scale);

        plain_vec.push_back(plain);
    }

    return plain_vec;
}

vector<Ciphertext> encrypt(
    SEALContext &context, const SecretKey &sk, const vector<double> &arr1, const vector<double> &arr2, double scale)
{
    vector<Plaintext> plain_vec = encode2vec(context, arr1, arr2, scale);
    Encryptor encryptor(context, sk);

    vector<Ciphertext> enc_vec(plain_vec.size());
    for (size_t i = 0; i < enc_vec.size(); i++)
    {
        encryptor.encrypt_symmetric(plain_vec[i], enc_vec[i]);
    }

    return enc_vec;
}

Ciphertext inner_product(SEALContext &context, const vector<int> &coeffs, const vector<Ciphertext> &enc_vec)
{
    auto &context_data = *context.first_context_data();
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    std::size_t coeff_count = enc_vec[0].poly_modulus_degree();

    size_t slot_count = coeff_count / 2;

    if (((coeffs.size() + slot_count - 1) / slot_count) != enc_vec.size())
    {
        throw invalid_argument("inner_product: input size incorrect");
    }

    size_t num = enc_vec.size();

    Ciphertext enc;
    enc.resize(context, parms.parms_id(), 2);
    enc.is_ntt_form() = true;
    enc.scale() = enc_vec.front().scale();

    for (size_t i = 0; i < num; i++)
    {
        size_t bgn = i * slot_count;
        size_t end = std::min(coeffs.size(), bgn + slot_count);

        vector<int> pod(coeff_count);
        pod[0] = coeffs[bgn];
        for (size_t j = bgn + 1; j < end; j++)
        {
            pod[coeff_count - (j - bgn)] = -coeffs[j];
        }

        Plaintext plain = encode_non_ntt(context, pod);

        if (!plain.is_zero())
        {
            if (enc.size() == 0)
            {
                util::dyadic_product_coeffmod(
                    enc_vec[i].data(0), plain.data(), coeff_count, coeff_modulus[0], enc.data(0));
                util::dyadic_product_coeffmod(
                    enc_vec[i].data(1), plain.data(), coeff_count, coeff_modulus[0], enc.data(1));
            }
            else
            {
                vector<uint64_t> values(coeff_count);
                util::dyadic_product_coeffmod(
                    enc_vec[i].data(0), plain.data(), coeff_count, coeff_modulus[0], values.data());
                util::add_poly_coeffmod(enc.data(0), values.data(), coeff_count, coeff_modulus[0], enc.data(0));
                util::dyadic_product_coeffmod(
                    enc_vec[i].data(1), plain.data(), coeff_count, coeff_modulus[0], values.data());
                util::add_poly_coeffmod(enc.data(1), values.data(), coeff_count, coeff_modulus[0], enc.data(1));
            }
        }
    }

    return enc;
}

// Ciphertext pick_sum(SEALContext &context, const vector<int> &coeffs, vector<Ciphertext> &enc_vec) {
//     auto &context_data = *context.first_context_data();
//     auto &parms = context_data.parms();
//     auto &coeff_modulus = parms.coeff_modulus();
//     size_t coeff_count = enc_vec[0].poly_modulus_degree();
//     const auto &ntt_tables = context_data.small_ntt_tables();
//
//     for (auto &enc: enc_vec) {
//         if (enc.is_ntt_form()) {
//             throw invalid_argument("pick_sum: should be in non_ntt_form");
//         }
//     }
//
//     for (auto &enc: enc_vec) {
//         if (enc.is_ntt_form()) {
//             util::inverse_ntt_negacyclic_harvey(enc, 1, ntt_tables);
//         }
//     }
//
//     auto pick_and_sum = [&](const vector<uint64_t> &arr0, const vector<uint64_t> &arr1, const vector<size_t> &index)
//     -> std::array<uint64_t, 2> {
//         uint64_t res0 = 0;
//         uint64_t res1 = 0;
//         for (size_t bgn = 0; bgn < index.size(); bgn += 256) {
//             size_t end = std::min(index.size(), bgn + 256);
//
//             __m256i sum_vec0 = _mm256_setzero_si256();
//             __m256i sum_vec1 = _mm256_setzero_si256();
//             size_t i = bgn;
//             for (; i + 4 <= end; i += 4) {
//                 __m256i indices = _mm256_loadu_si256((__m256i const*)&index[i]);
//                 __m256i values0 = _mm256_i64gather_epi64((long long int const*)arr0.data(), indices, 1);
//                 __m256i values1 = _mm256_i64gather_epi64((long long int const*)arr1.data(), indices, 1);
//                 sum_vec0 = _mm256_add_epi64(sum_vec0, values0);
//                 sum_vec1 = _mm256_add_epi64(sum_vec1, values1);
//             }
//
//             uint64_t sum0 = _mm256_extract_epi64(sum_vec0, 0) + _mm256_extract_epi64(sum_vec0, 1) +
//                             _mm256_extract_epi64(sum_vec0, 2) + _mm256_extract_epi64(sum_vec0, 3);
//
//             uint64_t sum1 = _mm256_extract_epi64(sum_vec1, 0) + _mm256_extract_epi64(sum_vec1, 1) +
//                             _mm256_extract_epi64(sum_vec1, 2) + _mm256_extract_epi64(sum_vec1, 3);
//
//             for (; i < end; i++) {
//                 sum0 += arr0[index[i]];
//                 sum1 += arr1[index[i]];
//             }
//             res0 = util::add_uint_mod(res0, sum0, ntt_tables[0].modulus());
//             res1 = util::add_uint_mod(res1, sum1, ntt_tables[0].modulus());
//         }
//
//         return {res0, res1};
//     };
//
//     size_t slot_count = coeff_count / 2;
//
//     Ciphertext enc;
//     enc.resize(context, parms.parms_id(), 2);
//     enc.is_ntt_form() = false;
//     enc.scale() = enc_vec.front().scale();
//
//     size_t num = enc_vec.size();
//     for (size_t i = 0; i < num; i++) {
//         size_t bgn = i * slot_count;
//         size_t end = std::min(coeffs.size(), bgn + slot_count);
//
//         vector<size_t> index;
//         for (size_t j = bgn; j < end; j++) {
//             if (coeffs[j] == 1) {
//                 index.push_back(j - bgn);
//             }
//         }
//
//         vector<uint64_t> ct0(coeff_count * 2 - 1);
//         vector<uint64_t> ct1(coeff_count * 2 - 1);
//
//         std::copy_n(enc_vec[i].data(0), coeff_count, ct0.data());
//         std::copy_n(enc_vec[i].data(1), coeff_count, ct1.data());
//
//         util::negate_poly_coeffmod(ct0.data() + 1, coeff_count-1, ntt_tables[0].modulus(), ct0.data() + coeff_count);
//         util::negate_poly_coeffmod(ct1.data() + 1, coeff_count-1, ntt_tables[0].modulus(), ct1.data() + coeff_count);
//
//         for (size_t j = 0; j < coeff_count; j++) {
//             vector<uint64_t> arr0(ct0.data() + j, ct0.data() + j + coeff_count);
//             vector<uint64_t> arr1(ct1.data() + j, ct1.data() + j + coeff_count);
//
//             auto [v0, v1] = pick_and_sum(arr0, arr1, index);
//             enc.data(0)[j] = util::add_uint_mod(enc.data(0)[j], v0, ntt_tables[0].modulus());
//             enc.data(1)[j] = util::add_uint_mod(enc.data(1)[j], v1, ntt_tables[0].modulus());
//         }
//     }
//
//     return enc;
// }

void sample_poly_normal(
    std::shared_ptr<seal::UniformRandomGenerator> rng, const seal::EncryptionParameters &parms, const double stddev,
    uint64_t *destination)
{
    auto coeff_modulus = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count = parms.poly_modulus_degree();

    if (util::are_close(stddev, 0.0))
    {
        util::set_zero_poly(coeff_count, coeff_modulus_size, destination);
        return;
    }

    seal::RandomToStandardAdapter engine(std::move(rng));
    util::ClippedNormalDistribution dist(
        0, stddev, util::global_variables::noise_distribution_width_multiplier * stddev);

    for (size_t i = 0; i < coeff_count; i++)
    {
        auto noise = static_cast<int64_t>(dist(engine));
        if (noise > 0)
        {
            for (size_t j = 0; j < coeff_modulus_size; j++)
            {
                destination[i + j * coeff_count] = static_cast<uint64_t>(noise);
            }
        }
        else if (noise < 0)
        {
            noise = -noise;
            for (size_t j = 0; j < coeff_modulus_size; j++)
            {
                destination[i + j * coeff_count] = coeff_modulus[j].value() - static_cast<uint64_t>(noise);
            }
        }
        else
        {
            for (size_t j = 0; j < coeff_modulus_size; j++)
            {
                destination[i + j * coeff_count] = 0;
            }
        }
    }
}

void encrypt_symmetric_zero(
    const SecretKey &secret_key, const SEALContext &context, parms_id_type parms_id, bool is_ntt_form, double stddev,
    Ciphertext &destination)
{
    // We use a fresh memory pool with `clear_on_destruction' enabled.
    seal::MemoryPoolHandle pool = seal::MemoryManager::GetPool(seal::mm_prof_opt::mm_force_new, true);

    const auto &context_data = *context.get_context_data(parms_id);
    const auto &parms = context_data.parms();
    const auto &coeff_modulus = parms.coeff_modulus();
    size_t coeff_modulus_size = coeff_modulus.size();
    size_t coeff_count = parms.poly_modulus_degree();
    uint64_t plain_mod = parms.plain_modulus().value();
    const auto &ntt_tables = context_data.small_ntt_tables();
    size_t encrypted_size = 2;

    destination.resize(context, parms_id, encrypted_size);

    auto rng_error = parms.random_generator()->create();
    shared_ptr<seal::UniformRandomGenerator> rng_ciphertext;
    rng_ciphertext = seal::Blake2xbPRNGFactory().create();

    // Generate ciphertext: (c[0], c[1]) = ([-(as+e)]_q, a)
    uint64_t *c0 = destination.data(0);
    uint64_t *c1 = destination.data(1);

    // Sample a uniformly at random
    // sample the NTT form directly
    util::sample_poly_uniform(rng_ciphertext, parms, c1);

    // Sample e <-- chi
    auto noise(util::allocate_poly(coeff_count, coeff_modulus_size, pool));
    sample_poly_normal(rng_error, parms, stddev, noise.get());

    // calculate -(a*s + e) (mod q) and store in c[0]
    for (size_t i = 0; i < coeff_modulus_size; i++)
    {
        util::dyadic_product_coeffmod(
            secret_key.data().data() + i * coeff_count, c1 + i * coeff_count, coeff_count, coeff_modulus[i],
            c0 + i * coeff_count);

        if (is_ntt_form)
        {
            // Transform the noise e into NTT representation.
            ntt_negacyclic_harvey(noise.get() + i * coeff_count, ntt_tables[i]);
        }
        util::multiply_poly_scalar_coeffmod(
            noise.get() + i * coeff_count, coeff_count, plain_mod, coeff_modulus[i], noise.get() + i * coeff_count);

        util::add_poly_coeffmod(
            noise.get() + i * coeff_count, c0 + i * coeff_count, coeff_count, coeff_modulus[i], c0 + i * coeff_count);

        util::negate_poly_coeffmod(c0 + i * coeff_count, coeff_count, coeff_modulus[i], c0 + i * coeff_count);
    }

    destination.parms_id() = parms_id;
    destination.is_ntt_form() = is_ntt_form;
    destination.scale() = 1.0;
}

inline void FMAU128(uint64_t *acc, const uint64_t *op0, const uint64_t *op1, const size_t len)
{
    unsigned long long wide[2];
    for (size_t i = 0; i < len; ++i, acc += 2)
    {
        seal::util::multiply_uint64(*op0++, *op1++, wide);
        uint64_t _wp[2] = { static_cast<uint64_t>(wide[0]), static_cast<uint64_t>(wide[1]) };
        auto carry = seal::util::add_uint(acc, _wp, 2, acc);

        if (carry != 0)
        {
            throw logic_error("FMAU128: overflow!");
        }
    }
}

inline uint64_t barrett_reduce_64_lazy(const uint64_t input, const seal::Modulus &modulus)
{
    // Reduces input using base 2^64 Barrett reduction
    // input must be at most 63 bits

    if (modulus.is_zero())
    {
        throw invalid_argument("barrett_reduce_64_lazy: invalid mdulus");
    }

    if ((input >> 63) != 0)
    {
        throw invalid_argument("barrett_reduce_64_lazy: input must be at most 63 bits");
    }

    unsigned long long tmp[2];
    const uint64_t *const_ratio = modulus.const_ratio().data();
    seal::util::multiply_uint64(input, const_ratio[1], tmp);

    // Barrett subtraction
    return input - tmp[1] * modulus.value();
}

inline uint64_t barrett_reduce_128_lazy(const uint64_t *input, const seal::Modulus &modulus)
{
    unsigned long long tmp1;
    unsigned long long tmp2[2];
    unsigned long long tmp3;
    unsigned long long carry;

    const uint64_t *const_ratio = modulus.const_ratio().data();

    // Multiply input and const_ratio
    // Round 1
    seal::util::multiply_uint64_hw64(input[0], const_ratio[0], &carry);

    seal::util::multiply_uint64(input[0], const_ratio[1], tmp2);
    tmp3 = tmp2[1] + seal::util::add_uint64(tmp2[0], carry, 0, &tmp1);

    // Round 2
    seal::util::multiply_uint64(input[1], const_ratio[0], tmp2);
    carry = tmp2[1] + seal::util::add_uint64(tmp1, tmp2[0], 0, &tmp1);

    // This is all we care about
    tmp1 = input[1] * const_ratio[1] + tmp3 + carry;

    // Barrett subtraction
    return input[0] - tmp1 * modulus.value();
}

struct FastMulMod
{
    uint64_t cnst, p;
    uint64_t cnst_shoup;

    explicit FastMulMod(uint64_t cnst, uint64_t p) : cnst(cnst), p(p)
    {
        uint64_t cnst_128[2]{ 0, cnst };
        uint64_t shoup[2];
        seal::util::divide_uint128_inplace(cnst_128, p, shoup);
        cnst_shoup = shoup[0]; // cnst_shoup = cnst * 2^64 / p
    }

    uint64_t lazy(uint64_t x) const
    {
        unsigned long long hw64;
        seal::util::multiply_uint64_hw64(x, cnst_shoup, &hw64);
        std::uint64_t q = static_cast<std::uint64_t>(hw64) * p;
        return (x * cnst - q);
    }

    inline uint64_t operator()(uint64_t x) const
    {
        uint64_t t = lazy(x);
        return t - ((p & -static_cast<uint64_t>(t < p)) ^ p);
    }
};

void key_switch(Ciphertext &ct, SEALContext &context1, const KSwitchKeys &switch_keys, size_t id)
{
    if (ct.size() == 0)
    {
        throw invalid_argument("key_switch: invalid ciphertext size");
    }
    if (!is_metadata_valid_for(ct, context1))
    {
        throw runtime_error("key_switch: invalid runtime");
    }
    if (ct.is_ntt_form())
    {
        throw logic_error("key_switch: require non_ntt ct");
    }

    auto mem_pool = MemoryManager::GetPool();
    auto key_cntxt = context1.key_context_data();
    auto ct_cntxt = context1.get_context_data(ct.parms_id());

    const auto &inv_qk_mod_q = key_cntxt->rns_tool()->inv_q_last_mod_q();
    const auto &nttTables = key_cntxt->small_ntt_tables();
    const size_t coeff_count = ct_cntxt->parms().poly_modulus_degree();
    const size_t coeff_modulus_size = ct.coeff_modulus_size();
    const size_t max_coeff_modulus_size = context1.first_context_data()->parms().coeff_modulus().size();

    const auto &skeys = switch_keys.data(id);

    size_t index = ct.size() - 1;

    Ciphertext enc;
    enc.resize(context1, ct.parms_id(), 2);
    enc.is_ntt_form() = ct.is_ntt_form();
    enc.scale() = ct.scale();

    // [ct'[0]]_{qj} <- \sum_{i} [skeys[i, 0]]_{qj} * [ct[1]]_{qi} mod qj
    // [ct'[1]]_{qj} <- \sum_{i} [skeys[i, 1]]_{qj} * [ct[1]]_{qi} mod qj
    // qj includes special primes(s), qi only loop over cipher moduli.

    util::Pointer<uint64_t> lazy_mul_poly[2] = { util::allocate_poly(coeff_count * 2, 1, mem_pool),
                                                 util::allocate_poly(coeff_count * 2, 1, mem_pool) };

    util::Pointer<uint64_t> spcl_rns_part[2] = { util::allocate_poly(coeff_count, 1, mem_pool),
                                                 util::allocate_poly(coeff_count, 1, mem_pool) };

    util::Pointer<uint64_t> nrml_rns_part = util::allocate_poly(coeff_count, 1, mem_pool);

    for (ssize_t j = coeff_modulus_size; j >= 0; --j)
    {
        const bool is_special = (static_cast<size_t>(j) >= coeff_modulus_size);
        const size_t rns_idx = is_special ? max_coeff_modulus_size + (j - coeff_modulus_size) : j;

        std::fill_n(lazy_mul_poly[0].get(), coeff_count * 2, 0);
        std::fill_n(lazy_mul_poly[1].get(), coeff_count * 2, 0);

        std::vector<uint64_t> tmp_rns(coeff_count);

        for (size_t i = 0; i < coeff_modulus_size; ++i)
        {
            const uint64_t *ct_ptr = ct.data(index) + i * coeff_count;

            if (nttTables[i].modulus().value() > nttTables[rns_idx].modulus().value())
            {
                // qi > qk
                auto mod_qj = nttTables[rns_idx].modulus();
                std::transform(ct_ptr, ct_ptr + coeff_count, tmp_rns.data(), [&mod_qj](uint64_t u) {
                    return seal::util::barrett_reduce_64(u, mod_qj);
                });
            }
            else
            {
                // qi < qk
                std::copy_n(ct_ptr, coeff_count, tmp_rns.data());
            }
            // ntt in [0, 4qi)
            util::ntt_negacyclic_harvey_lazy(tmp_rns.data(), nttTables[rns_idx]);
            const uint64_t *ct_qi_mod_qj = tmp_rns.data(); // [ct[l]_{qi}]_{qj}

            // [skeys[i, 0]]_{qj}
            const uint64_t *skeys0_qj = skeys[i].data().data(0) + rns_idx * coeff_count;
            // [skeys[i, 1]]_{qj}
            const uint64_t *skeys1_qj = skeys[i].data().data(1) + rns_idx * coeff_count;

            FMAU128(lazy_mul_poly[0].get(), skeys0_qj, ct_qi_mod_qj, coeff_count);
            FMAU128(lazy_mul_poly[1].get(), skeys1_qj, ct_qi_mod_qj, coeff_count);
        }

        // 2) Reduction and Rescale
        // 2-1) For special rns part, add (p-1)/2 and convert to the power-basis
        // 2-2) For normal rns part, compute qk^(1) (ct mod qi - ct mod qk) mod qi
        if (is_special)
        {
            for (int l : { 0, 1 })
            {
                auto acc_ptr = lazy_mul_poly[l].get();
                uint64_t *dst_ptr = spcl_rns_part[l].get();
                for (size_t d = 0; d < coeff_count; ++d, acc_ptr += 2)
                {
                    dst_ptr[d] = barrett_reduce_128_lazy(acc_ptr, nttTables[rns_idx].modulus());
                }

                util::inverse_ntt_negacyclic_harvey_lazy(dst_ptr, nttTables[rns_idx]);

                const uint64_t half = nttTables[rns_idx].modulus().value() >> 1U;
                SEAL_ITERATE(dst_ptr, coeff_count, [half, rns_idx, &nttTables](uint64_t &J) {
                    J = util::barrett_reduce_64(J + half, nttTables[rns_idx].modulus());
                });
            }
        }
        else
        {
            const Modulus &mod_qj = nttTables[j].modulus();
            const uint64_t qk = nttTables[max_coeff_modulus_size].modulus().value();
            const uint64_t qj = mod_qj.value();
            const uint64_t neg_half_mod = qj - util::barrett_reduce_64(qk >> 1U, mod_qj);
            uint64_t qj_lazy = qj << 1U; // some multiples of qi
            uint64_t inv_qk = inv_qk_mod_q[j].operand;

            std::vector<uint64_t> last_moduli(coeff_count);
            FastMulMod mulmod_s(inv_qk, qj);

            for (size_t l : { 0U, 1U })
            { // two cipher components
                const uint64_t *acc_ptr = lazy_mul_poly[l].get();
                uint64_t *dst_ptr = nrml_rns_part.get();
                // lazy reduce to [0, 2p)
                for (size_t d = 0; d < coeff_count; ++d, acc_ptr += 2)
                {
                    dst_ptr[d] = barrett_reduce_128_lazy(acc_ptr, nttTables[rns_idx].modulus());
                }

                // [0, 2p)
                util::inverse_ntt_negacyclic_harvey_lazy(dst_ptr, nttTables[j]);

                const uint64_t *last_moduli_ptr = spcl_rns_part[l].get();
                if (qk > qj)
                {
                    // Lazy add (p-1)/2, results in [0, 2p)
                    std::transform(
                        last_moduli_ptr, last_moduli_ptr + coeff_count, last_moduli.data(),
                        [neg_half_mod, &mod_qj](uint64_t u) {
                            return barrett_reduce_64_lazy(u + neg_half_mod, mod_qj);
                        });
                }
                else
                {
                    // Lazy add (p-1)/2, results in [0, 2p)
                    std::transform(
                        last_moduli_ptr, last_moduli_ptr + coeff_count, last_moduli.data(),
                        [neg_half_mod](uint64_t u) { return u + neg_half_mod; });
                }

                // qk^(-1) * ([ct]_qi - [ct]_qk) mod qi
                std::transform(
                    dst_ptr, dst_ptr + coeff_count, last_moduli.data(), dst_ptr,
                    [&mulmod_s, qj_lazy](uint64_t c, uint64_t v) { return mulmod_s(c + qj_lazy - v); });

                uint64_t *ans_ptr = enc.data(l) + j * coeff_count;
                uint64_t *ct_ptr = ct.data(l) + j * coeff_count;

                if (index == l)
                {
                    copy_n(dst_ptr, coeff_count, ans_ptr);
                }
                else
                {
                    util::add_poly_coeffmod(dst_ptr, ct_ptr, coeff_count, mod_qj, ans_ptr);
                }
            }
        } // normal rns part
    } // handle all the normal rns.

    ct = enc;
}

KSwitchKeys gen_switch_keys(SEALContext &context0, SEALContext &context1, const SecretKey &sk0, const SecretKey &sk1)
{
    auto &context_data0 = *context0.first_context_data();
    auto &parms0 = context_data0.parms();
    size_t coeff_count0 = parms0.poly_modulus_degree();
    const auto &ntt_tables0 = context_data0.small_ntt_tables();

    auto &context_data1 = *context1.key_context_data();
    auto &parms1 = context_data1.parms();
    size_t coeff_count1 = parms1.poly_modulus_degree();
    const auto &ntt_tables1 = context_data1.small_ntt_tables();
    const auto &key_modulus = parms1.coeff_modulus();

    if (parms0.coeff_modulus().size() != 1 || parms1.coeff_modulus().size() != 2 ||
        parms0.coeff_modulus()[0].value() != parms1.coeff_modulus()[0].value())
    {
        throw invalid_argument("gen_switch_keys: input parameters invalid");
    }

    vector<uint64_t> sk_values(coeff_count0);
    std::copy_n(sk0.data().data(), coeff_count0, sk_values.data());
    util::inverse_ntt_negacyclic_harvey(sk_values.data(), ntt_tables0[0]);

    size_t step = 2;
    size_t offset0 = coeff_count0 / step;
    size_t offset1 = coeff_count1 / step;
    KSwitchKeys switch_keys;

    switch_keys.data().resize(offset0);
    for (size_t k = 0; k < offset0; k++)
    {
        vector<uint64_t> temp(coeff_count1, 0);

        if (k == 0)
        {
            for (size_t i = 0; i < step; i++)
            {
                temp[i * offset1] = sk_values[i * offset0 + k];
            }
        }
        else
        {
            for (size_t i = 0; i < step; i++)
            {
                size_t kk = offset0 - k;
                temp[i * offset1] = sk_values[i * offset0 + kk];
            }
            uint64_t vv = temp[(step - 1) * offset1];
            for (size_t i = 1; i < step; i++)
            {
                temp[i * offset1] = temp[(i - 1) * offset1];
            }
            temp[0] = util::negate_uint_mod(vv, ntt_tables1[0].modulus());
        }

        const Modulus &qq = key_modulus[0];

        vector<PublicKey> destination(1);

        encrypt_symmetric_zero(
            sk1, context1, parms1.parms_id(), true, util::seal_he_std_parms_error_std_dev, destination[0].data());

        uint64_t factor = util::barrett_reduce_64(key_modulus.back().value(), qq);
        util::multiply_poly_scalar_coeffmod(temp.data(), coeff_count1, factor, qq, temp.data());

        util::ntt_negacyclic_harvey(temp.data(), ntt_tables1[0]);

        uint64_t *rns_ptr = destination[0].data().data(0);
        util::add_poly_coeffmod(rns_ptr, temp.data(), coeff_count1, qq, rns_ptr);

        destination[0].parms_id() = parms1.parms_id();
        switch_keys.data()[k] = destination;
    }

    switch_keys.parms_id() = parms1.parms_id();
    return switch_keys;
}

Ciphertext mlwe_pack(
    SEALContext &context0, SEALContext &context1, const vector<Ciphertext> &enc_vec, const KSwitchKeys &switchKeys)
{
    auto &context_data0 = *context0.first_context_data();
    auto &parms0 = context_data0.parms();
    size_t coeff_count0 = parms0.poly_modulus_degree();

    auto &context_data1 = *context1.first_context_data();
    auto &parms1 = context_data1.parms();
    size_t coeff_count1 = parms1.poly_modulus_degree();

    size_t step = 2;

    if (enc_vec.size() > (coeff_count1 / step))
    {
        throw invalid_argument("mlwe_pack: input size incorrect");
    }

    if (parms0.coeff_modulus().size() != 1 || parms1.coeff_modulus().size() != 1 ||
        parms0.coeff_modulus()[0].value() != parms1.coeff_modulus()[0].value())
    {
        throw invalid_argument("mlwe_pack: input parameters invalid");
    }

    size_t offset0 = coeff_count0 / step;
    size_t offset1 = coeff_count1 / step;

    vector<vector<uint64_t>> ct_v0s;
    vector<vector<uint64_t>> ct_v1s;

    for (const auto &enc : enc_vec)
    {
        vector<uint64_t> ct_v0(coeff_count0);
        vector<uint64_t> ct_v1(coeff_count0);
        std::copy_n(enc.data(0), coeff_count0, ct_v0.data());
        std::copy_n(enc.data(1), coeff_count0, ct_v1.data());
        if (enc.is_ntt_form())
        {
            const auto &ntt_tables0 = context_data0.small_ntt_tables();
            util::inverse_ntt_negacyclic_harvey(ct_v0.data(), ntt_tables0[0]);
            util::inverse_ntt_negacyclic_harvey(ct_v1.data(), ntt_tables0[0]);
        }
        ct_v0s.push_back(ct_v0);
        ct_v1s.push_back(ct_v1);
    }

    double scale = enc_vec.front().scale();

    vector<Ciphertext> enc_res(offset0);
    for (size_t k = 0; k < offset0; k++)
    {
        vector<uint64_t> ct0(coeff_count1, 0);
        vector<uint64_t> ct1(coeff_count1, 0);
        for (size_t j = 0; j < ct_v0s.size(); j++)
        {
            for (size_t i = 0; i < step; i++)
            {
                if (k == 0)
                {
                    ct0[i * offset1 + j] = ct_v0s[j][i * offset0];
                }
                ct1[i * offset1 + j] = ct_v1s[j][i * offset0 + k];
            }
        }

        enc_res[k].resize(context1, context1.first_parms_id(), 2);
        std::copy_n(ct0.data(), coeff_count1, enc_res[k].data(0));
        std::copy_n(ct1.data(), coeff_count1, enc_res[k].data(1));
        enc_res[k].scale() = scale;

        key_switch(enc_res[k], context1, switchKeys, k);
    }

    Evaluator evaluator(context1);
    Ciphertext encX;
    evaluator.add_many(enc_res, encX);
    evaluator.transform_to_ntt_inplace(encX);

    return encX;
}

vector<Ciphertext> histogram(
    SEALContext &context0, SEALContext &context1, const vector<vector<int>> &index_vec, size_t nrow,
    const vector<Ciphertext> &enc_vec, const KSwitchKeys &switchKeys)
{
    size_t ncol = index_vec.size();

    //    auto time_start = chrono::high_resolution_clock::now();

    vector<Ciphertext> enc_res(ncol);
    for (size_t j = 0; j < ncol; j++)
    {
        vector<int> coeff(nrow);
        for (auto &vv : index_vec[j])
        {
            if (vv >= nrow)
            {
                throw invalid_argument("histogram: input invalid");
            }
            coeff[vv] = 1;
        }

        enc_res[j] = inner_product(context0, coeff, enc_vec);
        //        enc_res[j] = pick_sum(context0, coeff, enc_vec);
    }

    //    auto time_end = chrono::high_resolution_clock::now();
    //    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    //    cout << ncol << " inner product takes " << time_diff.count() << " us" << endl;

    size_t coeff_count = enc_vec.front().poly_modulus_degree();

    vector<Ciphertext> encs((ncol + coeff_count - 1) / coeff_count);
    for (size_t bgn = 0; bgn < ncol; bgn += coeff_count)
    {
        size_t end = std::min(ncol, bgn + coeff_count);

        vector<Ciphertext> sub_encs(enc_res.begin() + bgn, enc_res.begin() + end);
        encs[bgn / coeff_count] = mlwe_pack(context0, context1, sub_encs, switchKeys);
    }

    return encs;
}

vector<double> decrypt(SEALContext &context, const SecretKey &sk, const vector<Ciphertext> &encs, size_t len)
{
    auto &context_data = *context.first_context_data();
    auto &parms = context_data.parms();
    std::size_t coeff_count = parms.poly_modulus_degree();
    auto ntt_tables = context_data.small_ntt_tables();

    if (encs.size() * coeff_count < (len * 2))
    {
        throw invalid_argument("decrypt: input size incorrect");
    }

    Decryptor decryptor(context, sk);

    size_t offset = coeff_count / 2;
    vector<double> res(len * 2);
    for (size_t bgn = 0; bgn < len; bgn += offset)
    {
        size_t end = std::min(len, bgn + offset);

        Plaintext plain;
        decryptor.decrypt(encs[bgn / offset], plain);

        util::inverse_ntt_negacyclic_harvey(plain.data(), ntt_tables[0]);

        uint64_t mod = parms.coeff_modulus().front().value();

        for (size_t l = 0; l < 2; l++)
        {
            for (size_t i = bgn; i < end; i++)
            {
                int64_t vv;
                size_t id = (i - bgn) + l * offset;
                if (plain[id] * 2 > mod)
                {
                    vv = static_cast<int64_t>(plain[id] - mod);
                }
                else
                {
                    vv = static_cast<int64_t>(plain[id]);
                }
                res[i + l * len] = static_cast<double>(vv) / plain.scale();
            }
        }
    }

    return res;
}

vector<double> histogram(const vector<vector<int>> &index_vec, const vector<double> &grad, const vector<double> &hess)
{
    size_t ncol = index_vec.size();
    size_t nrow = grad.size();

    if (grad.size() != hess.size())
    {
        throw invalid_argument("histogram: input size incorrect");
    }

    vector<double> real(ncol * 2, 0.0);
    for (size_t l = 0; l < ncol; l++)
    {
        const auto indexs = index_vec[l];
        for (auto index : indexs)
        {
            real[l] += grad[index];
            real[l + ncol] += hess[index];
        }
    }

    return real;
}

vector<double> gen_doubles(size_t n)
{
    std::uniform_real_distribution<double> dist(-1, 1);
    std::mt19937_64 gen(random_device{}());

    vector<double> res(n);
    for (auto &vv : res)
    {
        vv = dist(gen);
    }
    return res;
}

vector<int> gen_bools(size_t n)
{
    std::uniform_int_distribution<int> dist(0, 1);
    std::mt19937_64 gen(random_device{}());

    vector<int> res(n);
    for (auto &vv : res)
    {
        vv = dist(gen);
    }
    return res;
}

vector<vector<int>> gen_index_vec(size_t nrow, size_t ncol, size_t iv_num)
{
    //    ncol = ncol * split_num;
    //    vector<vector<int>> index_vec(ncol);
    //    for (size_t l = 0; l < ncol; l += split_num) {
    //        vector<double> vec = gen_doubles(nrow);
    //        for (auto &xx : vec) {
    //            xx = xx * 10;
    //        }
    //
    //        vector<double> splits = gen_doubles(split_num - 1);
    //        for (auto &xx : splits) {
    //            xx = xx * 10;
    //        }
    //        std::sort(splits.begin(), splits.end());
    //
    //        for (size_t i = 0; i < nrow; i++) {
    //            if (vec[i] < splits.front()) {
    //                index_vec[l].push_back(static_cast<int>(i));
    //            }
    //        }
    //
    //        for (size_t j = 1; j < splits.size(); j++) {
    //            for (size_t i = 0; i < nrow; i++) {
    //                if (splits[j - 1] <= vec[i] && vec[i] < splits[j]) {
    //                    index_vec[l + j].push_back(static_cast<int>(i));
    //                }
    //            }
    //        }
    //
    //        for (size_t i = 0; i < nrow; i++) {
    //            if (vec[i] >= splits.back()) {
    //                index_vec[l + split_num - 1].push_back(static_cast<int>(i));
    //            }
    //        }
    //    }

    uniform_int_distribution<int> dist(0, iv_num - 1);
    mt19937_64 gen(random_device{}());

    vector<vector<int>> index_vec(ncol * iv_num);
    for (size_t j = 0; j < ncol; j++)
    {
        for (size_t i = 0; i < nrow; i++)
        {
            int idx = dist(gen);
            index_vec[j * iv_num + idx].push_back(i);
        }
    }

    return index_vec;
}

// 18014398509309953 36028797018652673
void test()
{
    auto time_start = chrono::high_resolution_clock::now();
    EncryptionParameters parms(scheme_type::ckks);

    size_t coeff_count0 = 2048;
    parms.set_poly_modulus_degree(coeff_count0);
    parms.set_coeff_modulus({ Modulus(18014398509309953) });

    SEALContext context0(parms, true, sec_level_type::tc128);

    SecretKey sk0;
    KeyGenerator keygen0(context0);
    sk0 = keygen0.secret_key();

    double scale = std::pow(2, 30);

    size_t coeff_count1 = 4096;
    parms.set_poly_modulus_degree(coeff_count1);
    parms.set_coeff_modulus({ Modulus(18014398509309953), Modulus{ 36028797018652673 } });

    SEALContext context1(parms, true, sec_level_type::tc128);

    SecretKey sk1;
    KeyGenerator keygen1(context1);
    sk1 = keygen1.secret_key();

    KSwitchKeys switchKeys = gen_switch_keys(context0, context1, sk0, sk1);

    auto time_end = chrono::high_resolution_clock::now();
    auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "ckks initialize takes " << time_diff.count() << " us" << endl;

    size_t n = 100000;
    size_t m = 1000;
    size_t split_num = 4;
    vector<double> grad = gen_doubles(n);
    vector<double> hess = gen_doubles(n);
    vector<vector<int>> index_vec = gen_index_vec(n, m, split_num);

    time_start = chrono::high_resolution_clock::now();
    vector<Ciphertext> enc_vec = encrypt(context0, sk0, grad, hess, scale);
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "ckks encrypt grad and hess takes " << time_diff.count() << " us" << endl;

    time_start = chrono::high_resolution_clock::now();
    vector<Ciphertext> encs = histogram(context0, context1, index_vec, n, enc_vec, switchKeys);
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "ckks compute histogram takes " << time_diff.count() << " us" << endl;

    time_start = chrono::high_resolution_clock::now();
    vector<double> comp = decrypt(context1, sk1, encs, m * split_num);
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "ckks decrypt grad and hess takes " << time_diff.count() << " us" << endl;

    vector<double> real = histogram(index_vec, grad, hess);

    double max_err = 0.0;
    double avg_err = 0.0;
    for (size_t i = 0; i < m * split_num * 2; i++)
    {
        double diff = std::abs(comp[i] - real[i]);
        max_err = std::max(max_err, diff);
        avg_err = avg_err + diff;
    }
    cout << "max error: " << max_err << " avg error: " << (avg_err / (m * split_num * 2)) << endl;
}

int main()
{
    test();

    return 0;
}
