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
#include <chrono>
#include <map>
#include <iomanip>
#include "../agent/agent.h"
#include "../others/others.h"
#include "../hashtag/hashtag.h"
#include "../initialize/initialize.h"

void simulate_network(SimulationConfig &config, 
                      std::vector<agent> &agent_list, Hashtag_updater &hashtag_updater, int seed){
    auto tic = std::chrono::steady_clock::now();
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int> select_random_agent(0, agent_list.size() - 1);

    std::vector<int> posted_record;
    posted_record.reserve(config.one_day);

    PostResult post_result = PostResult::True;
    bool append_xt = false;
    bool append_post_record = false;
    bool append_new_users = false;   // [new-users]
    bool save_post = false;

    int day = 0;
    int trend_follow_false_cnt = 0;
    int trend_follow_false_cnt_max = -1;
    for (int time = 0; time < config.max_step; time++){
        /* post */
        agent_list[select_random_agent(mt)].post_network(agent_list, hashtag_updater, posted_record, time, post_result, save_post);
        if (post_result != PostResult::True){ // trend_follow_network failed{
            time --;
            trend_follow_false_cnt += 1;
            if(trend_follow_false_cnt >= 1000){
                std::cout << "trend_follow_network failed 1000 times, bad parameter!" << std::endl;
                break;
            }
            continue;
        }
        else{
            trend_follow_false_cnt_max = std::max(trend_follow_false_cnt_max, trend_follow_false_cnt);
            trend_follow_false_cnt = 0;
        }

        /* out put per day */
        if (((time+1)%config.one_day==0)){
            day += 1;
            if ((config.max_days - day) <= config.n_save_from_last_step) save_post = true;
            if ((config.max_days - day) < config.n_save_from_last_step){
                output_vector(posted_record, config.post_path, append_post_record);
                hashtag_updater.output_new_users(hashtag_updater.new_users, config.new_users_path, append_new_users);    // [new-users]
                append_new_users = true;
            }
            hashtag_updater.new_users.clear();           // [new-users] counts are per day
            posted_record.clear();
            append_post_record = true;
        }

        /* update lifetime */
        if ( (time + 1)%config.update_timing == 0 ){ // update every 10 minutes
            // Model 1 ("ou"): advance the OU process one 10-minute step.
            // Model 0 ("const_dk"): d_k is constant, nothing to update.
            if (config.lifetype == "ou") hashtag_updater.update_lifetime_ou_diff(time, config);
            
            if ((config.max_days - day) < config.n_save_from_last_step){
                hashtag_updater.output_xt(config.xt_path, append_xt);
                hashtag_updater.xt.clear();
                append_xt = true;
            }
            else hashtag_updater.xt.clear();
        }

        /* visiulize progress & modify hashtag */
        if (time%50000 == 0){ //5e4
            show_progress(tic, time, config.max_step, day, hashtag_updater.hashtag_num);
        }
    }

    std::cout << "\nMax trend_follow_network false cnt: " << trend_follow_false_cnt_max << std::endl;
}
