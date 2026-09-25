#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <fstream>
#include <vector>
#include <random>
#include <queue>
#include <iterator>
#include <string>
#include <algorithm>
#include <chrono>
#include <map>
#include <iomanip>
#include "config/config.h"
#include "agent/agent.h"
#include "others/others.h"
#include "hashtag/hashtag.h"
#include "initialize/initialize.h"
#include "models/models.h"
// using namespace std;

void simulation(SimulationConfig& config){
    auto tic = std::chrono::steady_clock::now();
    /************************ Load Prameters ************************/
    print_partition(60);
    std::cout << "Post output path: " << config.post_path << std::endl;
    std::cout << "max steps: " << config.max_step << std::endl;

    /************************ Initialize ************************/
    /* Hashtag */
    Hashtag_updater hashtag_updater = initialize_hashtags(config, config.seed + 1);   // [new-users] was `seed+1` with `seed` uninitialised

    /* Network */
    std::vector<int> network_links_vc = initialize_network(config.network);
    /* Agent */
    std::vector<agent> agent_list = initialize_agents(config, hashtag_updater, network_links_vc);
    std::cout << "Agent num: " << agent_list.size() << std::endl;
    /* post stream */

    /************************ Simulation ********************/
    if (config.change_interest == "network") simulate_network(config, agent_list, hashtag_updater, config.seed);
    else if (config.change_interest == "meanfield") std::cout << "under construction" << std::endl;
    else std::cout<< "wrong change insterest method" << std::endl;
    std::cout << std::endl;

    /************************ Output ************************/
    // Sampled d_k(r) trajectories for Fig. 9(a); empty unless lifetype == "ou".
    hashtag_updater.output_diffu_sample_series(config.diffu_sample_path, false);

    auto toc = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_time = toc - tic;
    std::cout << "Simulation time: " << elapsed_time.count()/60.0 << " minutes" << std::endl;
}

/************************ Main ************************/
int main(int argc, char *argv[])
{   
    if (argc == 1){
        std::cout << "Not enough input" << std::endl;
        return 0;
    }
    
    SimulationConfig config;
    config.model = std::stoi(argv[1]);
    config.agent_num = std::stoi(argv[2]);
    config.max_days = std::stoi(argv[3]);
    config.p_new = std::stod(argv[4]);

    config.lifetype = argv[5];
    config.network = argv[6];
    config.change_interest = argv[7];
    config.memory = std::stod(argv[8]);
    config.post_path = argv[9];

    config.sigma = std::stod(argv[10]);
    config.theta = std::stod(argv[11]);
    config.mu = std::stod(argv[12]);
    config.d0 = std::stod(argv[13]);
    config.one_hour = std::stoi(argv[14]);
    config.xt_path = argv[15];
    config.diffu_sample_path = argv[16];

    config.n_save_from_last_step = std::stoi(argv[17]);
    config.seed = std::stoi(argv[18]);
    config.new_users_path = argv[19];        // [new-users]

    config.one_day = config.one_hour * 24; // 24 * 93626 = 2247024 hashtags
    config.max_step = config.one_day * config.max_days;
    config.update_timing = config.one_hour / 6;
    std::cout << "update_timing: " << config.update_timing << std::endl;

    simulation(config);
}
