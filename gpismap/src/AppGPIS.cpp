#include "AppGPIS.h"
#include "covFnc.h"
#include <Eigen/Cholesky>
#include <chrono>
#include <thread>
#include <array>
#include <iostream>
#include <fstream>

#define SQRT_3  1.732051

void AppGPIS::reset() {
    nSamples = 0;
    trained = false;
    activeSet.clear();

    if (t != nullptr) {
        delete t;
        t = nullptr;
    }
    x.resize(0, 0);
    L.resize(0, 0);
    alpha.resize(0);
    prior_sig.resize(0);
    p_sig_valid.resize(0);
    gradflag.clear();
    obs_y.clear();

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

            f(k) = samples[k8 + 3];
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

        L = K_dyn.llt().matrixL();

        alpha = y;
        L.template triangularView<Eigen::Lower>().solveInPlace(alpha);
        L.transpose().template triangularView<Eigen::Upper>().solveInPlace(alpha);

        trained = true;
        // log
        /*std::ofstream logFile("C:\\Users\\zou\\source\\repos\\susiezou\\GPisMap2\\build\\training_logfile.txt", std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "Error opening the log file." << std::endl;
            return;
        }

        logFile << "This is a log message for training." << std::endl;
        logFile << "count:" << count << std::endl;
        logFile << "scale:" << param.scale << "; gradsigk: " << siggrad <<std::endl;
        logFile << "K_p Matrix:" << std::endl;
        logFile << K_p << std::endl;
        logFile << "K Matrix:" << std::endl;
        logFile << K << std::endl;
        logFile << "K_dyn Matrix:" << std::endl;
        logFile << K_dyn << std::endl;
        logFile << "alpha:" << alpha << std::endl;
        logFile.close();*/
    }
    return;
}

void AppGPIS::addNewMeas(const float* samples, float* p_sig, int N) {
    // create if not initialized
    if (t == nullptr) {
        t = new OcTree(Point3<float>(0.0, 0.0, 0.0));
    }
    nSamples = N;
    if (nSamples < 1)
        return;
    int num_elements = nSamples;
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int num_threads_to_use = num_threads;

    if (num_elements < num_threads) {
        num_threads_to_use = num_elements;
    }
    else {
        num_threads_to_use = num_threads;
    }
    int num_leftovers = num_elements % num_threads_to_use;
    int batch_size = num_elements / num_threads_to_use;
    int element_cursor = 0;
    for (int i = 0; i < num_leftovers; ++i) {
        std::thread thread_i = std::thread(&AppGPIS::addNewMeas_kernel,
            std::ref(*this),
            samples,p_sig,
            i,
            element_cursor,
            element_cursor + batch_size + 1);
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size + 1;
    }
    for (int i = num_leftovers; i < num_threads_to_use; ++i) {
        std::thread thread_i = std::thread(&AppGPIS::addNewMeas_kernel,
            std::ref(*this),
            samples,p_sig,
            i,
            element_cursor,
            element_cursor + batch_size);
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size;
    }

    for (auto& th : threads) {
        if (th.joinable())
            th.join();
    }

    return;
}

void AppGPIS::addNewMeas_kernel(const float* samples, float* psig, int thread_idx, int start_idx, int end_idx) 
{
    if (t == nullptr || nSamples < 1)
        return;

    for (int k = start_idx; k < end_idx; k++) {
        int k8 = k * 8;
        Point3<float> grad;
        grad.x = samples[k8 + 4];
        grad.y = samples[k8 + 5];
        grad.z = samples[k8 + 6];

        Point3<float> pt(samples[k8], samples[k8 + 1], samples[k8 + 2]);
        std::shared_ptr<Node3> p(new Node3(pt));
        std::unordered_set<OcTree*> vecInserted;

        bool succeeded = false;
        {
            std::lock_guard<std::mutex> lock(mux);
            if (!t->IsNotNew(p)) {
                succeeded = t->Insert(p, vecInserted);
                if (succeeded) {
                    if (t->IsRoot() == false) {
                        t = t->getRoot();
                    }
                }
            }
        }

        if ((succeeded == false) || !(vecInserted.size() > 0)) // if failed, then continue to test the next point
            continue;


        /////////////////////////////////////////////////////////////////
        // update the point
        p->updateData(samples[k8 + 3], psig[k], grad, samples[k8 + 7], NODE_TYPE::HIT);

        // supposed to have one element
        auto itv = vecInserted.cbegin();
        {
            std::lock_guard<std::mutex> lock(mux);
            activeSet.insert(*itv);
        }
        vecInserted.clear();
    }
    return;
}

void AppGPIS::updateGPs() {

    std::unordered_set<OcTree*> updateSet;

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    int num_threads_to_use = num_threads;

    for (auto it = activeSet.cbegin(); it != activeSet.cend(); it++) {

        Point3<float> ct = (*it)->getCenter();
        float l = (*it)->getHalfLength();
        AABB3 searchbb(ct.x, ct.y, ct.z, 2 * l);
        std::vector<OcTree*> qs;
        t->QueryNonEmptyLevelC(searchbb, qs);
        if (qs.size() > 0) {
            for (auto itq = qs.cbegin(); itq != qs.cend(); itq++) {
                updateSet.insert(*itq);
            }
        }
    }

    int num_elements = updateSet.size();
    if (num_elements < 1)
        return;

    std::vector<OcTree*> nodes_to_update;
    for (auto const& node : updateSet) {
        nodes_to_update.push_back(node);
    }

    if (num_elements < num_threads) {
        num_threads_to_use = num_elements;
    }
    else {
        num_threads_to_use = num_threads;
    }
    int num_leftovers = num_elements % num_threads_to_use;
    int batch_size = num_elements / num_threads_to_use;
    int element_cursor = 0;
    for (int i = 0; i < num_leftovers; ++i) {
        std::thread thread_i = std::thread(&AppGPIS::updateGPs_kernel,
            std::ref(*this),
            i,
            element_cursor,
            element_cursor + batch_size + 1,
            std::ref(nodes_to_update));
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size + 1;

    }
    for (int i = num_leftovers; i < num_threads_to_use; ++i) {
        std::thread thread_i = std::thread(&AppGPIS::updateGPs_kernel,
            std::ref(*this),
            i,
            element_cursor,
            element_cursor + batch_size,
            std::ref(nodes_to_update));
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size;
    }

    for (auto& th : threads) {
        if (th.joinable())
            th.join();
    }

    // clear active set once all the jobs for update are done.
    activeSet.clear();

    return;
}

void AppGPIS::updateGPs_kernel(int thread_idx,
    int start_idx,
    int end_idx,
    std::vector<OcTree*>& nodes_to_update) {
    std::vector<std::shared_ptr<Node3> > res;
    for (auto it = nodes_to_update.begin() + start_idx; it != nodes_to_update.begin() + end_idx; it++) {
        if ((*it) != nullptr) {
            Point3<float> ct = (*it)->getCenter();
            float l = (*it)->getHalfLength();
            AABB3 searchbb(ct.x, ct.y, ct.z, l * 2);
            res.clear();
            t->QueryRange(searchbb, res);
            if (res.size() > 0) {
                (*it)->InitGP(param.scale, param.noise);
                (*it)->UpdateGP_s(res);
            }
        }
    }

}


void AppGPIS::test(const float* samples, float* p_sig, int M, float* val, float* var) {

    if (!isTrained()) {
        std::cout << "GP Not trained!" << std::endl;
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

    std::ofstream logFile("C:\\Users\\zou\\source\\repos\\susiezou\\GPisMap2\\build\\test_logfile.txt", std::ios::app);
    if (!logFile.is_open()) {
       
        // regression
        EVectorX aa = K_dyn.transpose() * alpha;
        std::memcpy(val, aa.data(), aa.size() * sizeof(float));

        L.template triangularView<Eigen::Lower>().solveInPlace(K_dyn);

        K_dyn = K_dyn.array().pow(2);
        EVectorX v = K_dyn.colwise().sum();

        EVectorX vv = t_sig.array().pow(2) + param.noise * param.noise - v.array();
        std::memcpy(var, vv.data(), vv.size() * sizeof(float));
        return;
    }
    std::cerr << "Opening the log file." << std::endl;
    logFile << "This is a log message for testing." << std::endl;
    logFile << "K_row/col:" << K.rows() << "; " << K.cols() << "; K_dyn: " << K_dyn.rows() << "; " << K_dyn.cols() << std::endl;
    logFile << "K_p Matrix:" << std::endl;
    logFile << K_p.block(0, 0, 4, 4) << std::endl;
    logFile << "K Matrix:" << std::endl;
    logFile << K.block(0, 0, 4, 4) << std::endl;
    logFile << "K_dyn Matrix:" << std::endl;
    logFile << K_dyn.block(0, 0, 4, 4) << std::endl;
    // regression
    EVectorX aa = K_dyn.transpose() * alpha;
    std::memcpy(val, aa.data(), aa.size() * sizeof(float));
    logFile << "val:" << std::endl;
    for (int i = 0; i < aa.size(); ++i) {
        logFile << "Element " << i << ": " << val[i] << std::endl;
    }

    L.template triangularView<Eigen::Lower>().solveInPlace(K_dyn);
    
    K_dyn = K_dyn.array().pow(2);
    EVectorX v = K_dyn.colwise().sum();

    EVectorX vv = t_sig.array().pow(2) + param.noise * param.noise - v.array();
    std::memcpy(var, vv.data(), vv.size() * sizeof(float));
    logFile << "Result vv Matrix:" << std::endl;
    logFile << vv << std::endl;
    // log
    logFile.close();
    return;
}

bool AppGPIS::test(float* x, int leng, float* res) {
    if (x == nullptr || leng < 1)
        return false;

    int num_threads = std::thread::hardware_concurrency();
    int num_threads_to_use = num_threads;
    if (leng < num_threads) {
        num_threads_to_use = leng;
    }
    else {
        num_threads_to_use = num_threads;
    }
    std::vector<std::thread> threads;

    int num_leftovers = leng % num_threads_to_use;
    int batch_size = leng / num_threads_to_use;
    int element_cursor = 0;

    for (int i = 0; i < num_leftovers; ++i) {
        std::thread thread_i = std::thread(&AppGPIS::test_kernel_s,
            std::ref(*this),
            i,
            element_cursor,
            element_cursor + batch_size + 1,
            x, res);
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size + 1;

    }
    for (int i = num_leftovers; i < num_threads_to_use; ++i) {
        std::thread thread_i = std::thread(&AppGPIS::test_kernel_s,
            std::ref(*this),
            i,
            element_cursor,
            element_cursor + batch_size,
            x, res);
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size;
    }

    for (auto& th : threads) {
        if (th.joinable())
            th.join();
    }

    return true;
}

void AppGPIS::test_kernel(int thread_idx,
    int start_idx,
    int end_idx,
    float* x,
    float* res) {

    float var_thre = 0.5; // TO-DO

    for (int i = start_idx; i < end_idx; ++i) {

        int k3 = 3 * i;
        EVectorX xt(3);
        xt << x[k3], x[k3 + 1], x[k3 + 2];

        int k8 = 8 * i;

        // query Cs
        AABB3 searchbb(xt(0), xt(1), xt(2), (float)GPISMAP3_TREE_CLUSTER_HALF_LENGTH * 30.0);
        std::vector<OcTree*> octs;
        std::vector<float> sqdst;
        t->QueryNonEmptyLevelC(searchbb, octs, sqdst);

        res[k8 + 4] = 1.0 + param.noise; // variance of sdf value

        if (octs.size() == 1) {
            std::shared_ptr<OnGPIS> gp = octs[0]->getGP();
            if (gp != nullptr) {
                gp->testSinglePoint(xt, res[k8], &res[k8 + 1], &res[k8 + 4]);
            }
        }
        else if (sqdst.size() > 1) {
            // sort by distance
            std::vector<int> idx(sqdst.size());
            std::size_t n(0);
            std::generate(std::begin(idx), std::end(idx), [&] { return n++; });
            std::sort(std::begin(idx), std::end(idx), [&](int i1, int i2) { return sqdst[i1] < sqdst[i2]; });

            // get THE FIRST gp pointer
            std::shared_ptr<OnGPIS> gp = octs[idx[0]]->getGP();
            if (gp != nullptr) {
                gp->testSinglePoint(xt, res[k8], &res[k8 + 1], &res[k8 + 4]);
            }

            if (res[k8 + 4] > var_thre) {

                float f2[8];
                float grad2[8 * 3];
                float var2[8 * 4];

                var2[0] = res[k8 + 4];
                int numc = sqdst.size();
                if (numc > 3) numc = 3;
                bool need_wsum = true;

                for (int m = 0; m < (numc - 1); m++) {
                    int m_1 = m + 1;
                    int m3 = m_1 * 3;
                    int m4 = m_1 * 4;
                    gp = octs[idx[m_1]]->getGP();
                    gp->testSinglePoint(xt, f2[m_1], &grad2[m3], &var2[m4]);
                }

                if (need_wsum) {
                    f2[0] = res[k8];
                    grad2[0] = res[k8 + 1];
                    grad2[1] = res[k8 + 2];
                    grad2[2] = res[k8 + 3];
                    var2[1] = res[k8 + 5];
                    var2[2] = res[k8 + 6];
                    var2[3] = res[k8 + 7];
                    std::vector<int> idx(numc);
                    std::size_t n(0);
                    std::generate(std::begin(idx), std::end(idx), [&] { return n++; });
                    std::sort(std::begin(idx), std::end(idx), [&](int i1, int i2) { return var2[i1 * 4] < var2[i2 * 4]; });

                    if (var2[idx[0] * 4] < var_thre)
                    {
                        res[k8] = f2[idx[0]];
                        res[k8 + 1] = grad2[idx[0] * 3];
                        res[k8 + 2] = grad2[idx[0] * 3 + 1];
                        res[k8 + 3] = grad2[idx[0] * 3 + 2];

                        res[k8 + 4] = var2[idx[0] * 4];
                        res[k8 + 5] = var2[idx[0] * 4 + 1];
                        res[k8 + 6] = var2[idx[0] * 4 + 2];
                        res[k8 + 7] = var2[idx[0] * 4 + 3];
                    }
                    else {
                        float w1 = (var2[idx[0] * 4] - var_thre);
                        float w2 = (var2[idx[1] * 4] - var_thre);
                        float w12 = w1 + w2;

                        res[k8] = (w2 * f2[idx[0]] + w1 * f2[idx[1]]) / w12;
                        res[k8 + 1] = (w2 * grad2[idx[0] * 3] + w1 * grad2[idx[1] * 3]) / w12;
                        res[k8 + 2] = (w2 * grad2[idx[0] * 3 + 1] + w1 * grad2[idx[1] * 3 + 1]) / w12;
                        res[k8 + 3] = (w2 * grad2[idx[0] * 3 + 2] + w1 * grad2[idx[1] * 3 + 2]) / w12;

                        res[k8 + 4] = (w2 * var2[idx[0] * 4] + w1 * var2[idx[1] * 4]) / w12;
                        res[k8 + 5] = (w2 * var2[idx[0] * 4 + 1] + w1 * var2[idx[1] * 4 + 1]) / w12;
                        res[k8 + 6] = (w2 * var2[idx[0] * 4 + 2] + w1 * var2[idx[1] * 4 + 2]) / w12;
                        res[k8 + 7] = (w2 * var2[idx[0] * 4 + 3] + w1 * var2[idx[1] * 4 + 3]) / w12;
                    }
                }
            }
        }

    }

}

void AppGPIS::test_kernel_s(int thread_idx,
    int start_idx,
    int end_idx,
    float* x,
    float* res) {

    float var_thre = 0.5; // TO-DO

    for (int i = start_idx; i < end_idx; ++i) {

        int k4 = 4 * i;
        EVectorX xt(3);
        EVectorX sig2(1);
        xt << x[k4], x[k4 + 1], x[k4 + 2];
        sig2 << x[k4 + 3];

        int k8 = 8 * i;

        // query Cs
        AABB3 searchbb(xt(0), xt(1), xt(2), (float)GPISMAP3_TREE_CLUSTER_HALF_LENGTH * 30.0);
        std::vector<OcTree*> octs;
        std::vector<float> sqdst;
        t->QueryNonEmptyLevelC(searchbb, octs, sqdst);

        res[k8 + 4] = sig2(0); // variance of sdf value

        if (octs.size() == 1) {
            std::shared_ptr<OnGPIS> gp = octs[0]->getGP();
            if (gp != nullptr) {
                gp->testSinglePoint_s(xt, sig2, res[k8], &res[k8 + 1], &res[k8 + 4]);
            }
        }
        else if (sqdst.size() > 1) {
            // sort by distance
            std::vector<int> idx(sqdst.size());
            std::size_t n(0);
            std::generate(std::begin(idx), std::end(idx), [&] { return n++; });
            std::sort(std::begin(idx), std::end(idx), [&](int i1, int i2) { return sqdst[i1] < sqdst[i2]; });

            // get THE FIRST gp pointer
            std::shared_ptr<OnGPIS> gp = octs[idx[0]]->getGP();
            if (gp != nullptr) {
                gp->testSinglePoint_s(xt, sig2, res[k8], &res[k8 + 1], &res[k8 + 4]);
            }

            if (res[k8 + 4] > var_thre) {

                float f2[8];
                float grad2[8 * 3];
                float var2[8 * 4];

                var2[0] = res[k8 + 4];
                int numc = sqdst.size();
                if (numc > 3) numc = 3;
                bool need_wsum = true;

                for (int m = 0; m < (numc - 1); m++) {
                    int m_1 = m + 1;
                    int m3 = m_1 * 3;
                    int m4 = m_1 * 4;
                    gp = octs[idx[m_1]]->getGP();
                    gp->testSinglePoint_s(xt, sig2, f2[m_1], &grad2[m3], &var2[m4]);
                }

                if (need_wsum) {
                    f2[0] = res[k8];
                    grad2[0] = res[k8 + 1];
                    grad2[1] = res[k8 + 2];
                    grad2[2] = res[k8 + 3];
                    var2[1] = res[k8 + 5];
                    var2[2] = res[k8 + 6];
                    var2[3] = res[k8 + 7];
                    std::vector<int> idx(numc);
                    std::size_t n(0);
                    std::generate(std::begin(idx), std::end(idx), [&] { return n++; });
                    std::sort(std::begin(idx), std::end(idx), [&](int i1, int i2) { return var2[i1 * 4] < var2[i2 * 4]; });

                    if (var2[idx[0] * 4] < var_thre)
                    {
                        res[k8] = f2[idx[0]];
                        res[k8 + 1] = grad2[idx[0] * 3];
                        res[k8 + 2] = grad2[idx[0] * 3 + 1];
                        res[k8 + 3] = grad2[idx[0] * 3 + 2];

                        res[k8 + 4] = var2[idx[0] * 4];
                        res[k8 + 5] = var2[idx[0] * 4 + 1];
                        res[k8 + 6] = var2[idx[0] * 4 + 2];
                        res[k8 + 7] = var2[idx[0] * 4 + 3];
                    }
                    else {
                        float w1 = (var2[idx[0] * 4] - var_thre);
                        float w2 = (var2[idx[1] * 4] - var_thre);
                        float w12 = w1 + w2;

                        res[k8] = (w2 * f2[idx[0]] + w1 * f2[idx[1]]) / w12;
                        res[k8 + 1] = (w2 * grad2[idx[0] * 3] + w1 * grad2[idx[1] * 3]) / w12;
                        res[k8 + 2] = (w2 * grad2[idx[0] * 3 + 1] + w1 * grad2[idx[1] * 3 + 1]) / w12;
                        res[k8 + 3] = (w2 * grad2[idx[0] * 3 + 2] + w1 * grad2[idx[1] * 3 + 2]) / w12;

                        res[k8 + 4] = (w2 * var2[idx[0] * 4] + w1 * var2[idx[1] * 4]) / w12;
                        res[k8 + 5] = (w2 * var2[idx[0] * 4 + 1] + w1 * var2[idx[1] * 4 + 1]) / w12;
                        res[k8 + 6] = (w2 * var2[idx[0] * 4 + 2] + w1 * var2[idx[1] * 4 + 2]) / w12;
                        res[k8 + 7] = (w2 * var2[idx[0] * 4 + 3] + w1 * var2[idx[1] * 4 + 3]) / w12;
                    }
                }
            }
        }

    }

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

