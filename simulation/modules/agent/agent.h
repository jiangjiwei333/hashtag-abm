#pragma once
#include <vector>
#include <random>
#include <map>
#include <numeric>
#include <set>
#include <algorithm>

#include "../config/config.h"
#include "../hashtag/hashtag.h"
#include "../others/others.h"

enum class PostResult : uint8_t { 
    True, 
    Trend_false,
    Random_false,
    False
};

class agent
{
private:
public:
    int seed; // also equal to agent id

    std::vector<int> follow;
    std::vector<int> follower;
    std::deque<int> posted_list;     // own posts inside the memory window (hashtag ids)
    std::deque<int> posted_time;     // ... and their event times

    std::vector<int> time_line;
    std::vector<double> weights;
    std::vector<std::pair<int,int>> used_tags;   // [new-users] (tag, last day used), sorted by tag
    
    double p_new;
    double memory;
    
    std::mt19937 mt;
    std::uniform_real_distribution<double> gen_prob;
    const double memory_post;
    const int one_day;               // [new-users]

    /******************* Initialize ********************/
    agent(const SimulationConfig& config, double memory, Hashtag_updater &hashtag_updater, int ini_seed):
            p_new(config.p_new), memory(memory), memory_post(memory * config.one_hour),
            gen_prob(0.0, 1.0), seed(ini_seed), mt(ini_seed), one_day(config.one_day)
          {
            // initial hashtag: agent 0 creates one; every other agent creates a new one with
            int tag;
            // probability 0.1 and otherwise picks an existing one at random (all-new initial
            // hashtags would leave the system with too many singletons to converge)
            if (ini_seed == 0) tag = create_newtag(hashtag_updater, 0);
            else{
                if (gen_prob(mt) < 0.1) tag = create_newtag(hashtag_updater, 0);
                else{
                    std::uniform_int_distribution<int> random(0, hashtag_updater.hashtag_num - 1);
                    tag = random(mt);
                }
            }
            update_user(tag, 0);
            is_new_user(tag, 0);         // [new-users] the initial hashtag counts as used on day 0
            follow.reserve(1000);
            follower.reserve(1000);
          }

    /******************* User Activity *******************/
    void update_user(int tag, int time){
        posted_list.push_back(tag);
        posted_time.push_back(time);
        if ( (time - posted_time.front()) > (memory_post) ){ // drop the oldest post once it falls outside the memory window
            posted_list.pop_front();
            posted_time.pop_front();
        }
    }

    // [new-users] true when this post makes the agent a "new user" (paper: new adopter) of the
    // hashtag: it posts the hashtag today but did not post it on the previous day.
    // Also updates the stored last-use day. A second post of the same hashtag on the same day
    // returns false, so each user is counted at most once per hashtag per day.
    bool is_new_user(int tag, int day){
        auto it = std::lower_bound(used_tags.begin(), used_tags.end(), tag,
                                   [](const std::pair<int,int>& p, int t){ return p.first < t; });
        if (it == used_tags.end() || it->first != tag){
            used_tags.insert(it, {tag, day});
            return true;                                // never posted before -> new
        }
        if (it->second == day) return false;            // already posted today
        const bool is_new = (it->second < day - 1);     // false when it posted it yesterday
        it->second = day;
        return is_new;
    }

    /***************** Post *******************/
    // create a new hashtag
    int create_newtag(Hashtag_updater &hashtag_updater, int time){ // generate new hashtag
        int tag = hashtag_updater.hashtag_num;
        hashtag_updater.add_hashtag(tag, time);
        hashtag_updater.xt[tag] = 1;
        return tag;
    }
    
    // adopt one of the hashtags in followed users' recent posts, weighted by exposure x d_k
    int trend_follow_network_time_based(std::vector<agent> &agent_list, Hashtag_updater &hashtag_updater, int time){
        weights.clear();
        time_line.clear();
        for (auto &fol : follow){
            for (int i=agent_list[fol].posted_time.size()-1; i>=0; i--){            
                const int following_posted_time = agent_list[fol].posted_time[i];
                if ((double(time - following_posted_time) <= memory_post)) [[likely]] {
                    const int following_posted_tag = agent_list[fol].posted_list[i];
                    const double diffu = hashtag_updater.hashtag_diffu_vec[following_posted_tag];
                    if (diffu > 0.0){
                        time_line.push_back(following_posted_tag);
                        weights.push_back(diffu);
                    }
                }
                else break;
            }
        }
        if (time_line.size() == 0) return -1;
        else {
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            int random_tag = time_line[dist(mt)];
            hashtag_updater.xt[random_tag] += 1;
            return random_tag;
        }
    }

    void post_network(std::vector<agent> &agent_list, Hashtag_updater &hashtag_updater, std::vector<int> &posted_record, int time, PostResult &post_result, bool save_post){
        int tag;
        if (post_result == PostResult::Trend_false){ // previous attempt found nothing to follow: retry trend-following only
            tag = trend_follow_network_time_based(agent_list, hashtag_updater, time);
            if (tag == -1){
                post_result = PostResult::Trend_false;
                return;
            }
        }
        else{
            const double prob = gen_prob(mt);
            // create a new hashtag with probability p_new,
            if (prob < p_new) tag = create_newtag(hashtag_updater, time);
            // otherwise follow the trend
            else{
                tag = trend_follow_network_time_based(agent_list, hashtag_updater, time);
                if (tag == -1){
                    post_result = PostResult::Trend_false;
                    return;
                }
            }
        }

        if (is_new_user(tag, int(time / one_day))) hashtag_updater.new_users[tag] += 1;   // [new-users]
        if (save_post) posted_record.push_back(tag);
        update_user(tag, time);
        post_result = PostResult::True;
        return;
    }
};