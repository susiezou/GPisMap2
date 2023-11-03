#include "AppGPIS.h"
#include "covFnc.h"
#include <Eigen/Cholesky>

#define SQRT_3  1.732051

void AppGPIS::reset() {
    nSamples = 0;
    trained = false;

    x.resize(0, 0);
    L.resize(0, 0);
    alpha.resize(0);
    gradflag.clear();

    return;
}


void AppGPIS::train(const vecNode3& samples) {
    reset();

    int N = samples.size();
    int dim = 3;
    float noise = 0.02;

    if (N > 0) {
        nSamples = N;
        x = EMatrixX::Zero(dim, N);
        EMatrixX grad = EMatrixX::Zero(dim, N);
        EVectorX f = EVectorX::Zero(N);
        EVectorX sigx = EVectorX::Zero(N);
        EVectorX siggrad = EVectorX::Zero(N);

        prior_sig = EVectorX::Zero(N);
        p_sig_valid = EVectorX::Zero(N);

        gradflag.clear();
        gradflag.resize(N, 0.0);

        EMatrixX grad_valid(N, dim);

        int k = 0;
        int count = 0;
        for (auto it = samples.cbegin(); it != samples.cend(); it++, k++) {
            x(0, k) = (*it)->getPosX();
            x(1, k) = (*it)->getPosY();
            x(2, k) = (*it)->getPosZ();
            grad(0, k) = (*it)->getGradX();
            grad(1, k) = (*it)->getGradY();
            grad(2, k) = (*it)->getGradZ();
            f(k) = (*it)->getVal();
            sigx(k) = noise;
            prior_sig(k) = (*it)->getPosNoise();
            siggrad(k) = (*it)->getGradNoise();
            if (siggrad(k) > 0.1001 || (fabs(grad(0, k)) < 1e-6 && fabs(grad(1, k)) < 1e-6 && fabs(grad(2, k)) < 1e-6)) {
                gradflag[k] = 0.0;
                sigx(k) = 2.0;
            }
            else {
                gradflag[k] = 1.0;
                grad_valid(count, 0) = grad(0, k);
                grad_valid(count, 1) = grad(1, k);
                grad_valid(count, 2) = grad(2, k);
                p_sig_valid(count) = prior_sig(k);
                count++;
            }
        }
        grad_valid.conservativeResize(count, 3);
        p_sig_valid.conservativeResize(count);
        EMatrixX p_covq = prior_sig * p_sig_valid.transpose();
        EMatrixX p_valid = p_sig_valid * p_sig_valid.transpose();
        Eigen::MatrixXf K_p(N + count * dim, N + count * dim);
        K_p.block(N, N, count * dim, count * dim) = p_valid.replicate(dim, dim);
        K_p.block(0, 0, N, N) = prior_sig * prior_sig.transpose();
        K_p.block(0, N, N, count * dim) = p_covq.replicate(1, dim);
        K_p.block(N, 0, count * dim, N) = K_p.block(0, N, N, count * dim).transpose();

        EVectorX y(N + dim * count);
        y << f, grad_valid.col(0), grad_valid.col(1), grad_valid.col(2);
        EMatrixX K = matern32_sparse_deriv1(x, gradflag, param.scale, sigx, siggrad);
        EMatrixX K_dyn = K.cwiseProduct(K_p);

        // conta

        L = K_dyn.llt().matrixL();

        alpha = y;
        L.template triangularView<Eigen::Lower>().solveInPlace(alpha);
        L.transpose().template triangularView<Eigen::Upper>().solveInPlace(alpha);

        trained = true;

    }
    return;
}


void AppGPIS::train(const float* samples, float* p_sig, int N) {
    // samples = N * [x,y,z,mu,nx,ny,nz,noise_n]
    // p_sig = prior sigma
    reset();

    int dim = 3;

    if (N > 0) {
        nSamples = N;
        x = EMatrixX::Zero(dim, N);
        EMatrixX grad = EMatrixX::Zero(dim, N);
        EVectorX f = EVectorX::Zero(N);
        prior_sig = EVectorX::Zero(N);
        p_sig_valid = EVectorX::Zero(N);
        EVectorX sigx = EVectorX::Zero(N);
        EVectorX siggrad = EVectorX::Zero(N);

        gradflag.clear();
        gradflag.resize(N, 0.0);

        EMatrixX grad_valid(N, dim);

        int count = 0;
        for (int k = 0; k < N; k++) {
            int k8 = k * 8;
            x(0, k) = samples[k8];
            x(1, k) = samples[k8 + 1];
            x(2, k) = samples[k8 + 2];
            
            grad(0, k) = samples[k8 + 4];
            grad(1, k) = samples[k8 + 5];
            grad(2, k) = samples[k8 + 6];

            f(k) = -samples[k8 + 3];
            prior_sig(k) = p_sig[k];
            sigx(k) = param.noise;
            siggrad(k) = samples[k8 + 7];
            if (siggrad(k) > 0.1001 || (fabs(grad(0, k)) < 1e-6 && fabs(grad(1, k)) < 1e-6 && fabs(grad(2, k)) < 1e-6)) {
                gradflag[k] = 0.0;
                sigx(k) = 2.0;
            }
            else {
                gradflag[k] = 1.0;
                grad_valid(count, 0) = grad(0, k);
                grad_valid(count, 1) = grad(1, k);
                grad_valid(count, 2) = grad(2, k);
                p_sig_valid(count) = p_sig[k];
                count++;
            }
        }
        grad_valid.conservativeResize(count, 3);
        p_sig_valid.conservativeResize(count);
        EMatrixX p_covq = prior_sig * p_sig_valid.transpose();
        EMatrixX p_valid = p_sig_valid * p_sig_valid.transpose();
        Eigen::MatrixXf K_p(N + count * dim, N + count * dim);
        K_p.block(N, N, count*dim, count * dim) = p_valid.replicate(dim, dim);
        K_p.block(0, 0, N, N) = prior_sig * prior_sig.transpose();
        K_p.block(0, N, N, count * dim) = p_covq.replicate(1, dim);
        K_p.block(N, 0, count * dim, N) = K_p.block(0, N, N, count * dim).transpose();

        EVectorX y(N + dim * count);
        y << f, grad_valid.col(0), grad_valid.col(1), grad_valid.col(2);
        EMatrixX K = matern32_sparse_deriv1(x, gradflag, param.scale, sigx, siggrad);
        EMatrixX K_dyn = K.cwiseProduct(K_p);

        // conta
        

        L = K_dyn.llt().matrixL();

        alpha = y;
        L.template triangularView<Eigen::Lower>().solveInPlace(alpha);
        L.transpose().template triangularView<Eigen::Upper>().solveInPlace(alpha);

        trained = true;

    }
    return;
}


void AppGPIS::test(const float* samples, float* p_sig, int M, float* val, float* var) {

    if (!isTrained()) {
        return;
    }
    int dim = 3;
    EMatrixX xt = EMatrixX::Zero(dim, M);
    EVectorX t_sig = EVectorX::Zero(M);
    for (int k = 0; k < M; k++) {
        xt(0, k) = samples[k * dim];
        xt(1, k) = samples[k * dim + 1];
        xt(2, k) = samples[k * dim + 2];
        t_sig(k) = p_sig[k];
    }
    EMatrixX K = matern32_sparse_deriv1(x, gradflag, xt, param.scale);
    
    EMatrixX p_valid = p_sig_valid * t_sig.transpose();
    /*Eigen::MatrixXf K_p(K.rows(), K.cols());
    K_p.block(nSamples, 0, p_valid.rows() * dim, M*4) = p_valid.replicate(dim, dim+1);
    K_p.block(0, 0, nSamples, M * 4) = (prior_sig * t_sig.transpose()).replicate(1, dim+1);*/
    Eigen::MatrixXf K_p(K.rows(),M);
    K_p.block(nSamples, 0, p_valid.rows() * dim, M) = p_valid.replicate(dim, 1);
    K_p.block(0, 0, nSamples, M) = prior_sig * t_sig.transpose();
    EMatrixX K_dyn = K.block(0,0,K.rows(),M).cwiseProduct(K_p);

    // regression
    EVectorX aa = K_dyn.transpose() * alpha;
    val = aa.data();

    L.template triangularView<Eigen::Lower>().solveInPlace(K_dyn);

    K_dyn = K_dyn.array().pow(2);
    EVectorX v = K_dyn.colwise().sum();

    EVectorX vv = t_sig.array().pow(2) + param.noise * param.noise - v.array();
    var = vv.data();
   
    return;
}

void AppGPIS::testSinglePoint(const EVectorX& xt, float& val, float grad[], float var[])
{
    if (!isTrained())
        return;

    if (x.rows() != xt.size())
        return;

    EMatrixX K = matern32_sparse_deriv1(x, gradflag, xt, param.scale);

    EVectorX res = K.transpose() * alpha;
    val = res(0);
    if (res.size() == 3) {
        grad[0] = res(1);
        grad[1] = res(2);
    }
    else if (res.size() == 4) {
        grad[0] = res(1);
        grad[1] = res(2);
        grad[2] = res(3);
    }

    L.template triangularView<Eigen::Lower>().solveInPlace(K);
    K = K.array().pow(2);
    EVectorX v = K.colwise().sum();

    if (v.size() == 3) {
        var[0] = 1.01 - v(0);
        var[1] = three_over_scale + 0.1 - v(1);
        var[2] = three_over_scale + 0.1 - v(2);
    }
    else if (v.size() == 4) { // Noise param!
        var[0] = 1.001 - v(0);
        var[1] = three_over_scale + 0.001 - v(1);
        var[2] = three_over_scale + 0.001 - v(2);
        var[3] = three_over_scale + 0.001 - v(3);
    }

    return;
}

