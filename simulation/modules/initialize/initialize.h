#pragma once
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
#include <random>
#include <chrono>
#include <map>
#include <numeric>
#include "../config/config.h"
#include "../hashtag/hashtag.h"
#include "../others/others.h"


/************** Initialization *****************/
Hashtag_updater initialize_hashtags(const SimulationConfig& config, int seed){
    std::cout << "Initialize hashtags ..." << std::flush;
    Hashtag_updater hashtag_updater(config, seed);
    return hashtag_updater;
}

std::vector<agent> initialize_agents(const SimulationConfig& config, 
                                     Hashtag_updater &hashtag_updater, std::vector<int> &network_links_vc){
    /* Agent */
    std::cout << "Initialize agents ... " << std::flush;

    std::vector<agent> agent_list;
    agent_list.reserve(config.agent_num);
    for (int i = 0; i < config.agent_num; i++){
        agent_list.push_back(agent(config, config.memory, hashtag_updater, i));
    }
    std::cout<< "OK." <<std::endl;

    /* Agent Relationship */
    std::cout << "Add agent relationship ... " << std::flush;
    for (int i = 0; i < network_links_vc.size(); i++){
        if (i % 2 == 0){agent_list[network_links_vc[i]].follow.push_back(network_links_vc[i + 1]);}
        if (i % 2 == 1){agent_list[network_links_vc[i]].follower.push_back(network_links_vc[i - 1]);}
    }

    for (int i = 0; i < config.agent_num; i++){
        agent_list[i].follow.shrink_to_fit();
        agent_list[i].follower.shrink_to_fit();
        const int time_line_buf = int(config.memory)/24.0 * agent_list[i].follow.size(); // 48 / 24 * 5
        agent_list[i].time_line.reserve(time_line_buf); // 48 / 24 * 5
        agent_list[i].weights.reserve(time_line_buf);
    }
    
    std::cout<< "OK." <<std::endl;

    return agent_list;
}

// Load the user relationship network. Options used in the paper:
//   real                  empirical network (LCC of the retweet network)
//   real_to_random        complete random rewiring, same numbers of nodes and links   (Fig. 4e "random")
//   real_keep_in_degree   shuffle preserving in-degrees                              (Fig. 4e "keep in")
//   real_keep_out_degree  shuffle preserving out-degrees                             (Fig. 4e "keep out")
std::vector<int> initialize_network(std::string network){
    std::vector<int> network_links_vc;
    std::cout << "Loading Network(" << network << ")... " << std::flush;
    if (network == "none"){
        std::cout << "No network used." << std::endl;
    }
    else if (network == "real"){        
        std::string network_path = "./real_data/retweet_network/network_empirical.txt";
        read_lins_from_txt(network_path, network_links_vc);
    }
    else if (network == "real_to_random"){
        std::string network_path = "./real_data/retweet_network/network_shuffled_random.txt";
        read_lins_from_txt(network_path, network_links_vc);
    }
    else if (network == "real_keep_in_degree"){
        std::string network_path = "./real_data/retweet_network/network_shuffled_keep_in_degree.txt";
        read_lins_from_txt(network_path, network_links_vc);
    }
    else if (network == "real_keep_out_degree"){
        std::string network_path = "./real_data/retweet_network/network_shuffled_keep_out_degree.txt";
        read_lins_from_txt(network_path, network_links_vc);
    }
    else{ // check if loading successful
        std::cerr << "Unknown network type: " << network << std::endl;
        std::exit(1);
    }
    // A missing or empty edge list used to pass silently: every agent then has no one to
    // follow, trend following always fails and the run produces empty output files.
    if (network != "none" && network_links_vc.empty()){
        std::cerr << "\nerror: no links read for network '" << network
                  << "'. Check that the file exists under ./real_data/retweet_network/ "
                  << "and that the run is started from the folder containing simulation.sh."
                  << std::endl;
        std::exit(1);
    }
    return network_links_vc;
}


/************** Out Put *****************/

/************** Others *****************/
void show_progress(std::chrono::steady_clock::time_point tic, int time, int max_time, int day, int hashtag_numbers){
    std::cout << "\r" << std::string(80, ' ') << std::flush;
    auto toc = std::chrono::steady_clock::now();
    double progress = double(time) / double(max_time) * 100.0;
    double left = std::chrono::duration<double>(toc - tic).count() * (double(max_time) / double(time) - 1);
    std::cout << "\rSimulating ... Day " << day << " | "
                << std::fixed << std::setprecision(2)
                << std::setw(6)
                << progress << "%, left"
                << std::fixed << std::setprecision(2) << std::setw(6) << left << "s..."
                << "#hashtags ="  << std::setw(6) << hashtag_numbers
                << std::flush;
}