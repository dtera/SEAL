// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../examples.h"

using namespace std;
using namespace seal;

int main()
{
    /* 计算 （1.23 x 4.56）^ 2 +（7.89 x 10.11）^ 2 +（12.13 x 14.15）^ 2
     * 需要打包旋转 */
    /* 首先设置加密方案的参数 */
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 40, 40, 40, 40, 40 }));
    SEALContext context(parms);

    /* 生成密钥与加密、解密 */
    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    /* 使用 CKKSEncoder 批处理并编码 */
    CKKSEncoder encoder(context);
    auto slots = encoder.slot_count();
    /* 设置 scale 缩放的值 */
    double scale = pow(2.0, 40);

    /* 初始化向量 */
    vector<double> vx1(slots, 0.0), vx2(slots, 0.0), v1(slots, 0.0), v2(slots, 0.0);
    double real = 0;
    int n = slots / 16;
    for (int i = 0; i < n; i++)
    {
        vx1[i] = i + 1;
        vx2[i] = 2 * i + 1;
        double t = vx1[i] * vx2[i];
        if (i < n)
        {
            real += t * t;
        }
    }

    cout << "待编码的向量vx1:";
    print_vector(vx1);
    cout << "待编码的向量vx2:";
    print_vector(vx2);

    /* 编码并加密 */
    Plaintext plain_vx1, plain_vx2, pt1, pt2;
    Ciphertext encrypted_vx1, encrypted_vx2;
    encoder.encode(vx1, scale, plain_vx1);
    encoder.encode(vx2, scale, plain_vx2);
    encryptor.encrypt(plain_vx1, encrypted_vx1);
    encryptor.encrypt(plain_vx2, encrypted_vx2);

    decryptor.decrypt(encrypted_vx1, pt1);
    encoder.decode(pt1, v1);
    cout << "解密向量vx1:";
    print_vector(v1);
    decryptor.decrypt(encrypted_vx2, pt2);
    encoder.decode(pt2, v2);
    cout << "解密向量vx2:";
    print_vector(v2);

    Ciphertext add, sub;
    evaluator.add(encrypted_vx1, encrypted_vx2, add);
    decryptor.decrypt(add, pt1);
    encoder.decode(pt1, v1);
    cout << "解密向量vx1 + vx2:";
    print_vector(v1);
    evaluator.sub(encrypted_vx2, encrypted_vx1, sub);
    decryptor.decrypt(sub, pt2);
    encoder.decode(pt2, v2);
    cout << "解密向量vx2 - vx1:";
    print_vector(v2);

    /* 计算 （1.23 x 4.56）^ 2 +（7.89 x 10.11）^ 2 +（12.13 x 14.15）^ 2 */
    Ciphertext mul;
    evaluator.multiply(encrypted_vx1, encrypted_vx2, mul);
    evaluator.relinearize_inplace(mul, relin_keys);
    evaluator.rescale_to_next_inplace(mul);
    decryptor.decrypt(mul, pt1);
    encoder.decode(pt1, v1);
    cout << "解密向量vx1 * vx2:";
    print_vector(v1);
    evaluator.square_inplace(mul);
    evaluator.relinearize_inplace(mul, relin_keys);
    evaluator.rescale_to_next_inplace(mul);
    decryptor.decrypt(mul, pt2);
    encoder.decode(pt2, v2);
    cout << "解密向量(vx1 * vx2)^2:";
    print_vector(v2);

    /* 旋转并相加，计算结果 */
    Ciphertext total = mul;
    for (long i = 1; i < n; ++i)
    {
        Ciphertext tmp;
        evaluator.rotate_vector(mul, i, gal_keys, tmp);
        evaluator.add_inplace(total, tmp);
    }

    /* 解密 */
    /* 将密文解密解码至 vector */
    Plaintext total_plain;
    vector<double> result;
    decryptor.decrypt(total, total_plain);
    encoder.decode(total_plain, result);

    /* 打印结果 */
    cout << endl << "解密数组: ";
    print_vector(result); // 直接输出解密向量
    cout << "解密结果: " << result[0] << endl;
    cout << "实际结果: " << real << endl;

    return 0;
}
