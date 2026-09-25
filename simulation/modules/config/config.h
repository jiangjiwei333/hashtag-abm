// config.h
#ifndef CONFIG_H
#define CONFIG_H

#include <string>

struct SimulationConfig {
    int model;
    int agent_num;
    int max_days;
    int max_step;
    int n_save_from_last_step;
    double p_new;

    std::string lifetype;
    int update_timing;

    std::string network;
    std::string change_interest;
    double memory; // double is ok, later will be transfer to int

    std::string post_path;
    
    std::string xt_path;
    std::string diffu_sample_path;
    std::string new_users_path;        // [new-users] per-day counts of users who did not post the hashtag on the previous day

    int one_day; // defination of one day
    int one_hour;

    //ou_diff parameters
    double sigma;
    double theta;
    double mu;
    double d0;
    double dt;

    int seed;
};

#endif // CONFIG_H
