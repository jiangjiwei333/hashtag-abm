#pragma once
#include <map>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include <queue>

#include "../config/config.h" 
#include "../others/others.h"

class Hashtag_updater {
private:
    std::mt19937 mt;
public:
    // hashtag map
    int hashtag_num;

    // OU process parameters
    double mu;
    double theta;
    double sigma;
    double dt;
    double d0;
    std::vector<double> hashtag_diffu_vec;
    std::unordered_map<int, std::vector<double>> diffu_sample_map;
    
    std::unordered_map<int, int> xt;
    std::unordered_map<int, int> new_users;          // [new-users] tag -> users posting it today but not on the previous day
    std::string lifetype;

    std::uniform_real_distribution<double> uni;
    std::normal_distribution<double> normal;     // OU noise, N(0, sigma)

    Hashtag_updater(const SimulationConfig& config, int seed)
    :   mt(seed), hashtag_num(0), lifetype(config.lifetype),
        theta(config.theta), mu(config.mu), sigma(config.sigma), d0(config.d0), dt(config.dt),
        uni(0.0, 1.0)
        {
            hashtag_diffu_vec.reserve(int(config.p_new * config.max_step) + int(0.1 * config.agent_num) );
            normal = std::normal_distribution<double>(0.0, sigma);
        }

    /************* update hashtag map **************/
    void add_hashtag(int tag, int time) {
        double diffu_value;
        if (lifetype == "ou"){                  // Model 1: d_k(0) ~ U(0, d0), then OU
            diffu_value = uni(mt) * d0;
        }
        else if (lifetype == "const_dk"){       // Model 0: exposure only, d_k == 1
            diffu_value = 1.0;
            if (tag == 0) diffu_value = d0;     // special for tag 0
        }
        else{
            std::cerr << "Unknown lifetime type: " << lifetype
                      << " (expected \"ou\" or \"const_dk\")" << std::endl;
            std::exit(1);
        }
        hashtag_diffu_vec.push_back(diffu_value);
        // Record a small, deterministic sample of d_k(r) trajectories for Fig. 9(a).
        // Deterministic on purpose: the original `if (uni(mt) < 0.0001)` consumed a random
        // draw on every call and therefore shifted the hashtag-side random stream.
        if (tag % 10000 == 0) diffu_sample_map[tag].push_back(diffu_value);
        hashtag_num ++;
    }

    /************* update diffusion strength **************/
    void update_lifetime_ou_diff(int time, const SimulationConfig& config){
        for (int tag_index = 0; tag_index < hashtag_diffu_vec.size(); ++tag_index) {
            double drift = theta * (mu - hashtag_diffu_vec[tag_index]);
            double diffusion =  normal(mt);
            hashtag_diffu_vec[tag_index] += (drift + diffusion);
        }
        for (auto& [tag, series] : diffu_sample_map) {   // Fig. 9(a): sampled trajectories
            series.push_back(hashtag_diffu_vec[tag]);
        }
    }
    
    /***************** output *******************/
    void output_xt(std::string &path, bool append = false)
    {
        std::ofstream ofs;
        if (append) {
            ofs.open(path, std::ios::app);
        } else {
            ofs.open(path);
        }
        bool first = true;
        for (const auto& [tag, value] : xt) {
            if (!first) {
                ofs << ",";
            }
            ofs << tag << "," << value;
            first = false;
        }
        ofs << '\n';
        ofs.close();
    }

    // [new-users] same format as output_xt: "tag,count,tag,count,..." for the current day
    void output_new_users(const std::unordered_map<int, int>& counts, std::string &path, bool append = false)
    {
        std::ofstream ofs;
        if (append) ofs.open(path, std::ios::app);
        else        ofs.open(path);
        bool first = true;
        for (const auto& [tag, value] : counts) {
            if (!first) ofs << ",";
            ofs << tag << "," << value;
            first = false;
        }
        ofs << '\n';
        ofs.close();
    }

    void output_diffu_sample_series(std::string &path, bool append = false){
        std::ofstream ofs;
        if (append) {
            ofs.open(path, std::ios::app);
        } else {
            ofs.open(path);
        }
        for (const auto& [tag, values] : diffu_sample_map) {
            ofs << tag;
            for (const auto& value : values) {
                ofs << "," << value;
            }
            ofs << '\n';
        }
        ofs.close();
    }

};