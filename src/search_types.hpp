#pragma once
#include "base_state.hpp"
#include "search_params.hpp"
#include <vector>
#include <cstdint>
#include <functional>
#include <chrono>

class State;

struct RootUpdate {
    Move best_move;
    int score;
    int depth;
    int move_number;
    int total_moves;
};

struct SearchContext {
    uint64_t nodes = 0;
    int seldepth = 0;
    bool stop = false;
    ParamMap params;
    std::function<void(const RootUpdate&)> on_root_update;

    // Session timer
    double time_limit_ms = 0.0;  // 0 = no limit
    std::chrono::steady_clock::time_point search_start;

    void reset(){
        nodes = 0;
        seldepth = 0;
        search_start = std::chrono::steady_clock::now();
    }

    double elapsed_ms() const {
        using namespace std::chrono;
        return duration_cast<duration<double, std::milli>>(
            steady_clock::now() - search_start
        ).count();
    }

    // Returns true when the caller-set time limit has expired.
    bool time_up() const {
        return time_limit_ms > 0.0 && elapsed_ms() >= time_limit_ms;
    }
};

struct SearchResult {
    Move best_move;
    int score = 0;
    int depth = 0;
    int seldepth = 0;
    uint64_t nodes = 0;
    double time_ms = 0;
    std::vector<Move> pv;
};
