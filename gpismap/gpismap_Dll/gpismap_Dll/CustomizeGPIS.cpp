#include "CustomizeGPIS.h"
#include <chrono>
#include <thread>
#include "params.h"
#include <array>

static float Rtimes = (float)GPISMAP3_RTIMES;
static float C_leng = (float)GPISMAP3_TREE_CLUSTER_HALF_LENGTH;
tree_param OcTree::param = tree_param((float)GPISMAP3_TREE_MIN_HALF_LENGTH,
    (float)GPISMAP3_TREE_MAX_HALF_LENGTH,
    (float)GPISMAP3_TREE_INIT_ROOT_HALF_LENGTH,
    C_leng);

static inline bool isRangeValid(float r)
{
    return (r < GPISMAP3_MAX_RANGE) && (r > GPISMAP3_MIN_RANGE);
}

static inline float occ_test(float rinv, float rinv0, float a)
{
    return 2.0 * (1.0 / (1.0 + exp(-a * (rinv - rinv0))) - 0.5);
}

static inline float saturate(float val, float min_val, float max_val)
{
    return std::min(std::max(val, min_val), max_val);
}

static std::array<float, 9> quat2dcm(float q[4]) {
    std::array<float, 9> dcm;
    dcm[0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3];
    dcm[1] = 2.0 * (q[1] * q[2] + q[0] * q[3]);
    dcm[2] = 2.0 * (q[1] * q[3] - q[0] * q[2]);

    dcm[3] = 2.0 * (q[1] * q[2] - q[0] * q[3]);
    dcm[4] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3];
    dcm[5] = 2.0 * (q[0] * q[1] + q[2] * q[3]);

    dcm[6] = 2.0 * (q[1] * q[3] + q[0] * q[2]);
    dcm[7] = 2.0 * (q[2] * q[3] - q[0] * q[1]);
    dcm[8] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];

    return dcm;
}

CustomizeGPIS::CustomizeGPIS() :t(nullptr),
                                obs_numdata(0)
{
    init();
}

CustomizeGPIS::CustomizeGPIS(GPisMap3Param par) :t(nullptr),
                                                obs_numdata(0),
                                                setting(par)
{
    init();
}



CustomizeGPIS::~CustomizeGPIS()
{
    reset();
}

void CustomizeGPIS::init() {
    pose_tr.resize(3);
    pose_R.resize(9);
}

void CustomizeGPIS::reset() {

    activeSet.clear();

    if (t != nullptr) {
        delete t;
        t = nullptr;
    }

    obs_numdata = 0;

    return;
}


int CustomizeGPIS::getSampleCount() {
    if (t != nullptr)
        return t->getNodeCount();
    return 0;
}

bool CustomizeGPIS::getAllSamples(float* psamples, int dim, int leng, bool grad, bool var)
{
    if (t == nullptr || dim != 3)
        return false;

    std::vector<float> samples;
    bool res = getAllSamples(samples);
    int32_t data_size = leng * dim;
    if (grad)
        data_size = 2 * data_size;
    if (var)
        data_size += data_size / dim;

    if (res && (samples.size() == leng * dim)) {
        std::copy(samples.begin(), samples.end(), psamples);
        return true;
    }

    return false;
}

bool CustomizeGPIS::getAllSamples(std::vector<float>& samples, bool grad, bool var)
{
    if (t == nullptr)
        return false;

    std::vector<std::shared_ptr<Node3> > nodes;
    t->getAllChildrenNonEmptyNodes(nodes);

    samples.clear();
    for (auto const& node : nodes) {
        samples.push_back(node->getPosX());
        samples.push_back(node->getPosY());
        samples.push_back(node->getPosZ());
        if (grad) {
            samples.push_back(node->getGradX());
            samples.push_back(node->getGradY());
            samples.push_back(node->getGradZ());
        }
        if (var) {
            samples.push_back(node->getPosNoise());
            if (grad) {
                samples.push_back(node->getGradNoise());
            }
        }
    }
    return true;;
}

bool CustomizeGPIS::preprocData(float* dataz, float* datan, float* p_sig, int N)
{
    if (dataz == 0 || N < 1)
        return false;

    obs_valid_xyznormal.clear();
    obs_valid_xyzglobal.clear();
    obs_valid_dis.clear();
    obs_valid_psig.clear();
    obs_valid_nsig.clear();

    // pre-compute 3D cartesian every frame
    obs_numdata = 0;
    for (int k = 0; k < N; k++) {
        int k4 = k * 4;
        obs_valid_xyzglobal.push_back(dataz[k4]);
        obs_valid_xyzglobal.push_back(dataz[k4 + 1]);
        obs_valid_xyzglobal.push_back(dataz[k4 + 2]);
        obs_valid_dis.push_back(dataz[k4 + 3]);
        obs_valid_psig.push_back(p_sig[k]);

        obs_valid_xyznormal.push_back(datan[k4]);
        obs_valid_xyznormal.push_back(datan[k4 + 1]);
        obs_valid_xyznormal.push_back(datan[k4 + 2]);
        obs_valid_nsig.push_back(datan[k4 + 3]);

        obs_numdata++;
    }

    if (obs_numdata > 1)
        return true;

    return false;
}

void CustomizeGPIS::update(float* dataz, float* datan, float* p_sig, int N) {
    if (!preprocData(dataz, datan, p_sig, N))
        return;

    // Step 1
    if (true) {

        // Step 2
        updateMapPoints();

        // Step 3
        addNewMeas();

        // Step 4
        updateGPs();

    }
    return;
}

void CustomizeGPIS::updateMapPoints() {

    if (t != nullptr && gpo != nullptr) {
        AABB3 searchbb(pose_tr[0], pose_tr[1], pose_tr[2], range_obs_max);
        std::vector<OcTree*> oc;
        t->QueryNonEmptyLevelC(searchbb, oc);

        if (oc.size() > 0) {

            std::vector<std::shared_ptr<Node3> > nodes;
            float r2 = range_obs_max * range_obs_max;
            int k = 0;
            for (auto it = oc.cbegin(); it != oc.cend(); it++, k++) {

                Point3<float> ct = (*it)->getCenter();
                float l = (*it)->getHalfLength();
                float sqr_range = (ct.x - pose_tr[0]) * (ct.x - pose_tr[0]) + (ct.y - pose_tr[1]) * (ct.y - pose_tr[1]) + (ct.z - pose_tr[2]) * (ct.z - pose_tr[2]);

                if (sqr_range > (r2 + 2 * l * l)) { // out_of_range
                    continue;
                }

                std::map<OctChildType, Point3<float> > const& children_centers = (*it)->getAllChildrenCenter();
                for (auto const& child : children_centers) {
                    float x_loc = pose_R[0] * (child.second.x - pose_tr[0]) + pose_R[1] * (child.second.y - pose_tr[1]) + pose_R[2] * (child.second.z - pose_tr[2]);
                    float y_loc = pose_R[3] * (child.second.x - pose_tr[0]) + pose_R[4] * (child.second.y - pose_tr[1]) + pose_R[5] * (child.second.z - pose_tr[2]);
                    float z_loc = pose_R[6] * (child.second.x - pose_tr[0]) + pose_R[7] * (child.second.y - pose_tr[1]) + pose_R[8] * (child.second.z - pose_tr[2]);
                    if (z_loc > 0) {
                        float xv = x_loc / z_loc;
                        float yv = y_loc / z_loc;

                        bool within_angle = (xv > u_obs_limit[0]) && (xv < u_obs_limit[1]) && (yv > v_obs_limit[0]) && (yv < v_obs_limit[1]);
                        if (within_angle)
                            (*it)->getChildNonEmptyNodes(child.first, nodes);
                    }
                }
            }
            reEvalPoints(nodes);
        }

    }

    return;
}

void CustomizeGPIS::reEvalPoints(std::vector<std::shared_ptr<Node3> >& nodes) {
    int num_elements = nodes.size();
    if (num_elements < 1)
        return;

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
        std::thread thread_i = std::thread(&GPisMap3::reEvalPoints_kernel,
            std::ref(*this),
            i,
            element_cursor,
            element_cursor + batch_size + 1,
            std::ref(nodes));
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size + 1;

    }
    for (int i = num_leftovers; i < num_threads_to_use; ++i) {
        std::thread thread_i = std::thread(&GPisMap3::reEvalPoints_kernel,
            std::ref(*this),
            i,
            element_cursor,
            element_cursor + batch_size,
            std::ref(nodes));
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size;
    }

    for (auto& th : threads) {
        if (th.joinable())
            th.join();
    }
}



void CustomizeGPIS::reEvalPoints_kernel(int thread_idx,
    int start_idx,
    int end_idx,
    std::vector<std::shared_ptr<Node3> >& nodes) {

    // placeholders
    EMatrixX vu(2, 1);
    EVectorX rinv0(1);
    EVectorX var(1);
    float ang = 0.0;
    float rinv = 0.0;

    // moving average weight
    float w = 1.0 / 6.0;

    // For each point
    for (auto it = nodes.cbegin() + start_idx; it != nodes.cbegin() + end_idx; it++) {

        Point3<float> pos = (*it)->getPos();

        float x_loc = pose_R[0] * (pos.x - pose_tr[0]) + pose_R[1] * (pos.y - pose_tr[1]) + pose_R[2] * (pos.z - pose_tr[2]);
        float y_loc = pose_R[3] * (pos.x - pose_tr[0]) + pose_R[4] * (pos.y - pose_tr[1]) + pose_R[5] * (pos.z - pose_tr[2]);
        float z_loc = pose_R[6] * (pos.x - pose_tr[0]) + pose_R[7] * (pos.y - pose_tr[1]) + pose_R[8] * (pos.z - pose_tr[2]);

        if (z_loc < 0.0) {
            continue;
        }

        vu(0) = y_loc / z_loc;
        vu(1) = x_loc / z_loc;
        rinv = 1.0 / z_loc;

        gpo->test(vu, rinv0, var);

        // If unobservable, continue
        if (var(0) > setting.obs_var_thre) {
            continue;
        }

        float oc = occ_test(rinv, rinv0(0), z_loc * 30.0);

        // If unobservable, continue
        if (oc < -0.02) {
            continue;
        }

        // gradient in the local coord.
        Point3<float> grad = (*it)->getGrad();
        float grad_loc[3];
        grad_loc[0] = pose_R[0] * grad.x + pose_R[1] * grad.y + pose_R[2] * grad.z;
        grad_loc[1] = pose_R[3] * grad.x + pose_R[4] * grad.y + pose_R[5] * grad.z;
        grad_loc[2] = pose_R[6] * grad.x + pose_R[7] * grad.y + pose_R[8] * grad.z;

        /// Compute a new position
        // Iteratively move along the normal direction.
        float abs_oc = fabs(oc);
        float dx = setting.delx;
        float x_new[3] = { x_loc, y_loc, z_loc };
        float r_new = z_loc;
        for (int i = 0; i < 10 && abs_oc > 0.02; i++) { // TO-DO : set it as a parameter
            // move one step
            // (the direction is determined by the occupancy sign,
            //  the step size is heuristically determined accordint to iteration.)
            if (oc < 0) {
                x_new[0] += grad_loc[0] * dx;
                x_new[1] += grad_loc[1] * dx;
                x_new[2] += grad_loc[2] * dx;
            }
            else {
                x_new[0] -= grad_loc[0] * dx;
                x_new[1] -= grad_loc[1] * dx;
                x_new[2] -= grad_loc[2] * dx;
            }

            // test the new point
            vu(0) = y_loc / z_loc;
            vu(1) = x_loc / z_loc;
            r_new = z_loc;
            gpo->test(vu, rinv0, var);

            if (var(0) > setting.obs_var_thre)
                break;
            else {
                float oc_new = occ_test(1.0 / (r_new), rinv0(0), r_new * 30.0);
                float abs_oc_new = fabs(oc_new);

                if (abs_oc_new < 0.02 || oc < -0.02) // TO-DO : set it as a parameter
                    break;
                else if (oc * oc_new < 0.0)
                    dx = 0.5 * dx; // TO-DO : set it as a parameter
                else
                    dx = 1.1 * dx; // TO-DO : set it as a parameter

                abs_oc = abs_oc_new;
                oc = oc_new;
            }
        }

        // Compute its gradient and uncertainty
        float Xperturb[6] = { 1.0, -1.0, 0.0, 0.0, 0.0, 0.0 };
        float Yperturb[6] = { 0.0, 0.0, 1.0, -1.0, 0.0, 0.0 };
        float Zperturb[6] = { 0.0, 0.0, 0.0, 0.0, 1.0, -1.0 };
        float occ[6] = { -1.0,-1.0,-1.0,-1.0,-1.0,-1.0 };
        float occ_mean = 0.0;
        float r0_mean = 0.0;
        float r0_sqr_sum = 0.0;

        for (int i = 0; i < 6; i++) {
            Xperturb[i] = x_new[0] + setting.delx * Xperturb[i];
            Yperturb[i] = x_new[1] + setting.delx * Yperturb[i];
            Zperturb[i] = x_new[2] + setting.delx * Zperturb[i];

            vu(0) = Yperturb[i] / Zperturb[i];
            vu(1) = Xperturb[i] / Zperturb[i];
            r_new = Zperturb[i];
            gpo->test(vu, rinv0, var);

            if (var(0) > setting.obs_var_thre)
            {
                break;
            }
            occ[i] = occ_test(1.0 / r_new, rinv0(0), r_new * 30.0);
            occ_mean += w * occ[i];
            float r0 = 1.0 / rinv0(0);
            r0_sqr_sum += r0 * r0;
            r0_mean += w * r0;
        }

        if (var(0) > setting.obs_var_thre) {
            continue;
        }

        Point3<float> grad_new_loc, grad_new;

        grad_new_loc.x = (occ[0] - occ[1]) / setting.delx;
        grad_new_loc.y = (occ[2] - occ[3]) / setting.delx;
        grad_new_loc.z = (occ[4] - occ[5]) / setting.delx;
        float norm_grad_new = std::sqrt(grad_new_loc.x * grad_new_loc.x + grad_new_loc.y * grad_new_loc.y + grad_new_loc.z * grad_new_loc.z);

        if (norm_grad_new < 1e-3) { // uncertainty increased
            (*it)->updateNoise(2.0 * (*it)->getPosNoise(), 2.0 * (*it)->getGradNoise());
            continue;
        }

        float r_var = r0_sqr_sum / 5.0 - r0_mean * r0_mean * 6.0 / 5.0;
        r_var /= setting.delx;
        float noise = 100.0;
        float grad_noise = 1.0;
        if (norm_grad_new > 1e-6) {
            grad_new_loc.x = grad_new_loc.x / norm_grad_new;
            grad_new_loc.y = grad_new_loc.y / norm_grad_new;
            grad_new_loc.z = grad_new_loc.z / norm_grad_new;
            noise = setting.min_position_noise * saturate(r_new * r_new, 1.0, noise);
            grad_noise = saturate(std::fabs(occ_mean) + r_var, setting.min_grad_noise, grad_noise);
        }
        else {
            noise = setting.min_position_noise * noise;
        }

        float dist = std::sqrt(x_new[0] * x_new[0] + x_new[1] * x_new[1] + x_new[2] * x_new[2]);
        float view_ang = std::max(-(x_new[0] * grad_new_loc.x + x_new[1] * grad_new_loc.y + x_new[2] * grad_new_loc.z) / dist, (float)1e-1);
        float view_ang2 = view_ang * view_ang;
        float view_noise = setting.min_position_noise * ((1.0 - view_ang2) / view_ang2);

        float temp = noise;
        noise += view_noise + abs_oc;
        grad_noise = grad_noise + 0.1 * view_noise;

        // local to global coord.
        Point3<float> pos_new;
        pos_new.x = pose_R[0] * x_new[0] + pose_R[3] * x_new[1] + pose_R[6] * x_new[2] + pose_tr[0];
        pos_new.y = pose_R[1] * x_new[0] + pose_R[4] * x_new[1] + pose_R[7] * x_new[2] + pose_tr[1];
        pos_new.z = pose_R[2] * x_new[0] + pose_R[5] * x_new[1] + pose_R[8] * x_new[2] + pose_tr[2];
        grad_new.x = pose_R[0] * grad_new_loc.x + pose_R[3] * grad_new_loc.y + pose_R[6] * grad_new_loc.z;
        grad_new.y = pose_R[1] * grad_new_loc.x + pose_R[4] * grad_new_loc.y + pose_R[7] * grad_new_loc.z;
        grad_new.z = pose_R[2] * grad_new_loc.x + pose_R[5] * grad_new_loc.y + pose_R[8] * grad_new_loc.z;

        float noise_old = (*it)->getPosNoise();
        float grad_noise_old = (*it)->getGradNoise();

        float pos_noise_sum = (noise_old + noise);
        float grad_noise_sum = (grad_noise_old + grad_noise);

        // Now, update
        if (noise_old > 0.5 || grad_noise_old > 0.6) {
            ;
        }
        else {
            // Position update
            pos_new.x = (noise * pos.x + noise_old * pos_new.x) / pos_noise_sum;
            pos_new.y = (noise * pos.y + noise_old * pos_new.y) / pos_noise_sum;
            pos_new.z = (noise * pos.z + noise_old * pos_new.z) / pos_noise_sum;
            float dist = 0.5 * std::sqrt((pos.x - pos_new.x) * (pos.x - pos_new.x) + (pos.y - pos_new.y) * (pos.y - pos_new.y) + (pos.z - pos_new.z) * (pos.z - pos_new.z));

            // Normal update
            Point3<float> axis; // cross product
            axis.x = grad_new.y * grad.z - grad_new.z * grad.y;
            axis.y = -grad_new.x * grad.z + grad_new.z * grad.x;
            axis.z = grad_new.x * grad.y - grad_new.y * grad.x;
            float ang = acos(grad_new.x * grad.x + grad_new.y * grad.y + grad_new.z * grad.z);
            ang = ang * noise / pos_noise_sum;
            // rotvect
            float q[4] = { 1.0, 0.0, 0.0, 0.0 };
            if (ang > 1 - 6) {
                q[0] = cos(ang / 2.0);
                float sina = sin(ang / 2.0);
                q[1] = axis.x * sina;
                q[2] = axis.y * sina;
                q[3] = axis.z * sina;
            }
            // quat 2 dcm
            std::array<float, 9> Rot = quat2dcm(q);

            grad_new.x = Rot[0] * grad.x + Rot[1] * grad.y + Rot[2] * grad.z;
            grad_new.y = Rot[3] * grad.x + Rot[4] * grad.y + Rot[5] * grad.z;
            grad_new.z = Rot[6] * grad.x + Rot[7] * grad.y + Rot[8] * grad.z;

            // Noise update
            grad_noise = std::min((float)1.0, std::max(grad_noise * grad_noise_old / grad_noise_sum + dist, setting.map_noise_param));
            noise = std::max((noise * noise_old / pos_noise_sum + dist), setting.map_noise_param);
        }

        {
            std::lock_guard<std::mutex> lock(mux);
            // remove
            t->Remove(*it);
        }

        if (noise > 1.0 && grad_noise > 0.61) {
            continue;
        }
        else {
            // try inserting
            std::shared_ptr<Node3> p(new Node3(pos_new));
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

            if ((succeeded == false) || !(vecInserted.size() > 0)) {// if failed, then continue to test the next point
                continue;
            }

            // update the point
            p->updateData(-setting.fbias, noise, grad_new, grad_noise, NODE_TYPE::HIT);

            // supposed to have one element
            auto itv = vecInserted.cbegin();
            {
                std::lock_guard<std::mutex> lock(mux);
                activeSet.insert(*itv);
            }
            vecInserted.clear();
        }
    }
    return;
}

void CustomizeGPIS::addNewMeas() {
    // create if not initialized
    if (t == nullptr) {
        t = new OcTree(Point3<float>(0.0, 0.0, 0.0));
    }
    evalPoints();
    return;
}

void CustomizeGPIS::evalPoints() {
    if (obs_numdata < 1)
        return;
    int num_elements = obs_numdata;
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
        std::thread thread_i = std::thread(&CustomizeGPIS::evalPoints_kernel,
            std::ref(*this),
            i,
            element_cursor,
            element_cursor + batch_size + 1);
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size + 1;
    }
    for (int i = num_leftovers; i < num_threads_to_use; ++i) {
        std::thread thread_i = std::thread(&CustomizeGPIS::evalPoints_kernel,
            std::ref(*this),
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
}
// zqq modify
void CustomizeGPIS::evalPoints_kernel(int thread_idx,
                                        int start_idx,
                                        int end_idx) {

    if (t == nullptr || obs_numdata < 1)
        return;

    float w = 1.0 / 6.0;

    // For each point
    for (int k = start_idx; k < end_idx; k++) {
        int k2 = 2 * k;
        int k3 = 3 * k;

        /////////////////////////////////////////////////////////////////
        // Try inserting
        Point3<float> pt(obs_valid_xyzglobal[k3], obs_valid_xyzglobal[k3 + 1], obs_valid_xyzglobal[k3 + 2]);
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
        Point3<float> grad;

        grad.x = obs_valid_xyznormal[k3];
        grad.y = obs_valid_xyznormal[k3+1];
        grad.z = obs_valid_xyznormal[k3+2];
        p->updateData(-obs_valid_dis[k], obs_valid_psig[k], grad, obs_valid_nsig[k], NODE_TYPE::HIT);

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

void CustomizeGPIS::updateGPs_kernel(int thread_idx,
                                    int start_idx,
                                    int end_idx,
                                    std::vector<OcTree*>& nodes_to_update) {
    std::vector<std::shared_ptr<Node3> > res;
    for (auto it = nodes_to_update.begin() + start_idx; it != nodes_to_update.begin() + end_idx; it++) {
        if ((*it) != nullptr) {
            Point3<float> ct = (*it)->getCenter();
            float l = (*it)->getHalfLength();
            AABB3 searchbb(ct.x, ct.y, ct.z, l * Rtimes);
            res.clear();
            t->QueryRange(searchbb, res);
            if (res.size() > 0) {
                (*it)->InitGP(setting.map_scale_param, setting.map_noise_param);
                (*it)->UpdateGP(res);
            }
        }
    }

}

void CustomizeGPIS::updateGPs() {

    std::unordered_set<OcTree*> updateSet;

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    int num_threads_to_use = num_threads;

    for (auto it = activeSet.cbegin(); it != activeSet.cend(); it++) {

        Point3<float> ct = (*it)->getCenter();
        float l = (*it)->getHalfLength();
        AABB3 searchbb(ct.x, ct.y, ct.z, Rtimes * l);
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
        std::thread thread_i = std::thread(&CustomizeGPIS::updateGPs_kernel,
            std::ref(*this),
            i,
            element_cursor,
            element_cursor + batch_size + 1,
            std::ref(nodes_to_update));
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size + 1;

    }
    for (int i = num_leftovers; i < num_threads_to_use; ++i) {
        std::thread thread_i = std::thread(&CustomizeGPIS::updateGPs_kernel,
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

void CustomizeGPIS::test_kernel(int thread_idx,
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
        AABB3 searchbb(xt(0), xt(1), xt(2), C_leng * 3.0);
        std::vector<OcTree*> octs;
        std::vector<float> sqdst;
        t->QueryNonEmptyLevelC(searchbb, octs, sqdst);

        res[k8 + 4] = 1.0 + setting.map_noise_param; // variance of sdf value

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

bool CustomizeGPIS::test(float* x, int dim, int leng, float* res) {
    if (x == nullptr || dim != mapDimension || leng < 1)
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
        std::thread thread_i = std::thread(&CustomizeGPIS::test_kernel,
            std::ref(*this),
            i,
            element_cursor,
            element_cursor + batch_size + 1,
            x, res);
        threads.push_back(std::move(thread_i));
        element_cursor += batch_size + 1;

    }
    for (int i = num_leftovers; i < num_threads_to_use; ++i) {
        std::thread thread_i = std::thread(&CustomizeGPIS::test_kernel,
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