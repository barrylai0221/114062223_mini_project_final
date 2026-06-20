#include <utility>
#include <vector>
#include <algorithm>
#include <cstring>
#include <climits>
#include <chrono>
#include "state.hpp"
#include "minimax.hpp"


/*============================================================
 * Session Timer (file-scoped, set at start of each search)
 *============================================================*/
static double g_time_limit_ms = 0.0;
static std::chrono::steady_clock::time_point g_search_start;

static bool internal_time_up(){
    if(g_time_limit_ms <= 0.0) return false;
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
        steady_clock::now() - g_search_start
    ).count() >= g_time_limit_ms;
}


/*============================================================
 * Transposition Table
 *
 * 1<<18 = 262 144 entries x 16 bytes = 4 MB.
 * Persists across search() calls (cleared only on startup).
 * Hash collision is tolerated: the flag==TT_NONE guard and
 * full-hash comparison provide sufficient protection.
 *============================================================*/
enum TTFlag : uint8_t { TT_NONE = 0, TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3 };

struct TTEntry {
    uint64_t hash    = 0;
    int32_t  score   = 0;
    uint8_t  depth   = 0;
    TTFlag   flag    = TT_NONE;
    uint8_t  from_sq = 0;
    uint8_t  to_sq   = 0;
}; // 16 bytes

static constexpr int TT_SIZE = 1 << 18;
static constexpr int TT_MASK = TT_SIZE - 1;
static TTEntry tt[TT_SIZE];

static uint8_t sq_encode(int r, int c){ return (uint8_t)(r * BOARD_W + c); }
static Move sq_decode(uint8_t from, uint8_t to){
    return {{(size_t)(from / BOARD_W), (size_t)(from % BOARD_W)},
            {(size_t)(to   / BOARD_W), (size_t)(to   % BOARD_W)}};
}


/*============================================================
 * File-scoped search tables
 *
 * Killer and history tables.  Reset at the start of each
 * search() call.  Single-threaded search only.
 *============================================================*/
static constexpr int MAX_SEARCH_PLY = 64;
static constexpr int HIST_SZ        = 64; // >= BOARD_H * BOARD_W

struct SearchTables {
    Move killers[MAX_SEARCH_PLY][2]; // [ply][slot]
    int  history[HIST_SZ][HIST_SZ];  // [from_sq][to_sq]

    void reset(){
        for(int i = 0; i < MAX_SEARCH_PLY; i++)
            killers[i][0] = killers[i][1] = Move{};
        memset(history, 0, sizeof(history));
    }
} g_tables;


/*============================================================
 * Helpers
 *============================================================*/

/* MVV-LVA + TT move + killer + history score for move ordering.
 * tt_move: best move from a previous TT hit (Move{} if none). */
static int score_move(
    const Move& m, const State* state, int ply,
    const MMParams& p, const Move& tt_move
){
    // TT move: always search first
    if(p.use_tt && m == tt_move) return 30000;

    auto& [from, to] = m;
    int opp      = 1 - state->player;
    int victim   = state->piece_at(opp,           (int)to.first,   (int)to.second);
    int attacker = state->piece_at(state->player, (int)from.first, (int)from.second);

    if(victim){
        // Capture: MVV-LVA
        return 20000 + PIECE_VALUES[victim] * 10 - PIECE_VALUES[attacker];
    }
    int k = std::min(ply, MAX_SEARCH_PLY - 1);
    if(p.use_killers){
        if(m == g_tables.killers[k][0]) return 9000;
        if(m == g_tables.killers[k][1]) return 8000;
    }
    if(p.use_history){
        int fsq = (int)from.first * BOARD_W + (int)from.second;
        int tsq = (int)to.first   * BOARD_W + (int)to.second;
        return g_tables.history[fsq][tsq];
    }
    return 0;
}

/* Count own pieces; used to suppress null-move in lean endgames. */
static int count_own_pieces(const State* state){
    int cnt = 0;
    for(int r = 0; r < BOARD_H; r++)
        for(int c = 0; c < BOARD_W; c++)
            if(state->piece_at(state->player, r, c)) cnt++;
    return cnt;
}


/*============================================================
 * quiescence  (file-scoped helper)
 *============================================================*/
static int quiescence(
    State* state, int q_depth, GameHistory& history, int ply,
    SearchContext& ctx, const MMParams& p, int alpha, int beta
){
    ctx.nodes++;
    if(ctx.stop) return 0;

    if(state->legal_actions.empty() && state->game_state == UNKNOWN)
        state->get_legal_actions();

    if(state->game_state == WIN)  return P_MAX - ply;  // current player can capture king = wins
    if(state->game_state == DRAW) return 0;

    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    if(stand_pat >= beta)  return stand_pat;
    if(stand_pat > alpha)  alpha = stand_pat;
    if(q_depth <= 0)       return alpha;

    for(auto& action : state->legal_actions){
        auto& [from, to]   = action;
        auto& [to_r, to_c] = to;
        if(state->piece_at(1 - state->player, (int)to_r, (int)to_c) == 0) continue;

        State* next = (State*)state->next_state(action);
        bool   same = next->same_player_as_parent();
        int raw = quiescence(next, q_depth - 1, history, ply + 1, ctx, p,
                              same ? alpha : -beta, same ? beta : -alpha);
        int score = same ? raw : -raw;
        delete next;
        if(score >= beta)  return score;
        if(score > alpha)  alpha = score;
    }
    return alpha;
}


/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax with:
 *   * Alpha-beta pruning
 *   * Transposition table (probe + store)
 *   * PVS  (Principal Variation Search)
 *   * LMR  (Late Move Reduction)
 *   * Killer heuristic
 *   * History heuristic
 *   * Null Move Pruning
 *   * Futility Pruning at depth 1
 *   * Quiescence Search at the horizon
 *   * Session timer (periodic node check, honours ctx.time_limit_ms)
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta,
    bool was_null
){
    ctx.nodes++;

    if((ctx.nodes & 511) == 0 && internal_time_up()) ctx.stop = true;

    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

    /* === Lazy move generation === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN)
        state->get_legal_actions();

    /* === Terminal checks === */
    if(state->game_state == WIN)  return P_MAX - ply;  // current player can capture king = wins
    if(state->game_state == DRAW) return 0;

    /* === Repetition check === */
    int rep_score;
    if(state->check_repetition(history, rep_score)) return rep_score;

    /* === Transposition Table probe ===
     * Done before history.push() so early returns leave the stack clean. */
    TTEntry& tte = tt[state->hash() & TT_MASK];
    Move tt_move{};
    bool tt_hit = (tte.hash == state->hash() && tte.flag != TT_NONE);
    if(tt_hit){
        tt_move = sq_decode(tte.from_sq, tte.to_sq);
        if(p.use_tt && (int)tte.depth >= depth){
            int ts = tte.score;
            if(tte.flag == TT_EXACT)              return ts;
            if(tte.flag == TT_LOWER && ts >= beta) return ts;
            if(tte.flag == TT_UPPER && ts <= alpha) return ts;
        }
    }

    history.push(state->hash());
    int original_alpha = alpha; // needed to classify the TT flag after the loop

    /* === Horizon === */
    if(depth <= 0){
        int score = p.use_quiescence
            ? quiescence(state, p.q_depth, history, ply, ctx, p, alpha, beta)
            : state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        history.pop(state->hash());
        return score;
    }

    /* === Null Move Pruning ===
     * Conditions: feature on, parent not null, depth sufficient,
     * not in a mating window, side has >2 pieces (avoid zugzwang). */
    if(p.use_null_move && !was_null && !ctx.stop &&
       depth >= p.null_move_r + 1 &&
       beta < P_MAX - 1 && count_own_pieces(state) > 2)
    {
        State* null_st = (State*)state->create_null_state();
        if(null_st && null_st->game_state != WIN){
            int null_raw = eval_ctx(null_st, depth - 1 - p.null_move_r,
                                     history, ply + 1, ctx, p,
                                     -beta, -beta + 1, /*was_null=*/true);
            delete null_st;
            if(-null_raw >= beta){
                history.pop(state->hash());
                return beta; // null move cutoff
            }
        } else {
            delete null_st;
        }
    }

    /* === Futility Pruning setup (depth == 1) ===
     * Compute static eval once; if it already beats beta return early (stand-pat). */
    int static_eval = INT_MIN;
    if(p.use_futility && depth == 1){
        static_eval = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        // Stand-pat: the side to move can "do nothing" and take this score.
        if(static_eval > alpha) alpha = static_eval;
        if(alpha >= beta){
            history.pop(state->hash());
            return static_eval;
        }
    }

    /* === Move ordering ===
     * Priority: TT move (30 000) > captures/MVV-LVA (20 000+) >
     *           killers (9 000/8 000) > history (0–7 999) */
    std::vector<std::pair<int, Move>> scored;
    scored.reserve(state->legal_actions.size());
    for(auto& m : state->legal_actions)
        scored.push_back({score_move(m, state, ply, p, tt_move), m});
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    /* === Negamax loop === */
    int best_score = M_MAX;
    Move best_move = scored.empty() ? Move{} : scored[0].second;
    int  move_index = 0;

    for(auto& [ms, action] : scored){
        if(ctx.stop) break;

        auto& [from, to] = action;
        bool is_capture  = state->piece_at(1 - state->player,
                                            (int)to.first, (int)to.second) != 0;
        int  k           = std::min(ply, MAX_SEARCH_PLY - 1);
        bool is_killer   = p.use_killers &&
                           (action == g_tables.killers[k][0] ||
                            action == g_tables.killers[k][1]);

        /* === Futility Pruning (depth == 1, quiet non-killer moves only) ===
         * If static_eval + margin can't beat alpha, skip this move. */
        if(p.use_futility && depth == 1 && !is_capture && !is_killer &&
           static_eval != INT_MIN && static_eval + p.futility_margin <= alpha)
        {
            move_index++;
            continue;
        }

        State* next = (State*)state->next_state(action);
        bool   same = next->same_player_as_parent();
        int    score;

        if(move_index == 0){
            /* ── PV move: full window, full depth ── */
            int raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p,
                                same ? alpha : -beta, same ? beta : -alpha);
            score = same ? raw : -raw;

        } else {
            /* ── LMR: progressive depth reduction for late quiet moves ── */
            int R = 0;
            if(p.use_lmr && !is_capture && !is_killer &&
               move_index >= p.lmr_full_moves && depth >= p.lmr_min_depth)
            {
                R = 1 + (move_index - p.lmr_full_moves) / 4;
            }

            bool do_nw = p.use_pvs || R > 0;

            if(do_nw){
                /* Null-window probe (at possibly reduced depth via LMR) */
                int nw_depth = std::max(0, depth - 1 - R);
                int raw_nw   = eval_ctx(next, nw_depth, history, ply + 1, ctx, p,
                                         same ? alpha       : -(alpha + 1),
                                         same ? (alpha + 1) :  -alpha);
                score = same ? raw_nw : -raw_nw;

                if(!ctx.stop && score > alpha){
                    /* Undo LMR: re-search at full depth with null window */
                    if(R > 0){
                        raw_nw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p,
                                           same ? alpha       : -(alpha + 1),
                                           same ? (alpha + 1) :  -alpha);
                        score = same ? raw_nw : -raw_nw;
                    }
                    /* Full-window re-search if inside (alpha, beta) */
                    if(!ctx.stop && score > alpha && score < beta){
                        int raw_full = eval_ctx(next, depth - 1, history, ply + 1, ctx, p,
                                                 same ? alpha : -beta,
                                                 same ? beta  : -alpha);
                        score = same ? raw_full : -raw_full;
                    }
                }
            } else {
                /* PVS and LMR both off: plain full-window search */
                int raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p,
                                    same ? alpha : -beta, same ? beta : -alpha);
                score = same ? raw : -raw;
            }
        }

        delete next;

        if(score > best_score){
            best_score = score;
            best_move  = action;
        }
        if(score > alpha) alpha = score;

        if(p.use_ab && alpha >= beta){
            /* Beta cutoff: update killers and history for quiet moves */
            if(!is_capture){
                if(p.use_killers){
                    g_tables.killers[k][1] = g_tables.killers[k][0];
                    g_tables.killers[k][0] = action;
                }
                if(p.use_history){
                    int fsq = (int)from.first * BOARD_W + (int)from.second;
                    int tsq = (int)to.first   * BOARD_W + (int)to.second;
                    g_tables.history[fsq][tsq] += depth * depth;
                    if(g_tables.history[fsq][tsq] > 7999)
                        g_tables.history[fsq][tsq] = 7999;
                }
            }
            break;
        }
        move_index++;
    }

    /* === Transposition Table store ===
     * Only write if the search completed (not interrupted). */
    if(p.use_tt && !ctx.stop){
        TTFlag store_flag;
        if(best_score <= original_alpha) store_flag = TT_UPPER;
        else if(best_score >= beta)      store_flag = TT_LOWER;
        else                             store_flag = TT_EXACT;

        // Replace if: empty slot, same hash (update), or shallower entry
        if(tte.flag == TT_NONE || tte.hash == state->hash() ||
           (int)tte.depth <= depth)
        {
            tte.hash    = state->hash();
            tte.score   = best_score;
            tte.depth   = (uint8_t)std::min(depth, 255);
            tte.flag    = store_flag;
            tte.from_sq = sq_encode((int)best_move.first.first,
                                     (int)best_move.first.second);
            tte.to_sq   = sq_encode((int)best_move.second.first,
                                     (int)best_move.second.second);
        }
    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — search
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();       // resets nodes, seldepth
    g_tables.reset();  // fresh killers + history (TT persists across calls)

    // Start session timer (TimeLimit param set by ubgi.cpp for movetime mode)
    g_time_limit_ms  = (double)param_int(ctx.params, "TimeLimit", 0);
    g_search_start   = std::chrono::steady_clock::now();

    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size())
        state->get_legal_actions();

    // Probe TT for a root best-move hint (carry over from previous depth)
    TTEntry& root_tte = tt[state->hash() & TT_MASK];
    Move root_tt_move{};
    if(root_tte.hash == state->hash() && root_tte.flag != TT_NONE)
        root_tt_move = sq_decode(root_tte.from_sq, root_tte.to_sq);

    // Order root moves
    std::vector<std::pair<int, Move>> root_scored;
    root_scored.reserve(state->legal_actions.size());
    for(auto& m : state->legal_actions)
        root_scored.push_back({score_move(m, state, 0, p, root_tt_move), m});
    std::sort(root_scored.begin(), root_scored.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    int best_score  = M_MAX - 1;
    int alpha       = M_MAX;
    int beta        = P_MAX;
    int move_index  = 0;
    int total_moves = (int)root_scored.size();

    for(auto& [ms, action] : root_scored){
        if(ctx.stop) break;

        State* next = (State*)state->next_state(action);
        bool   same = next->same_player_as_parent();

        int raw = eval_ctx(next, depth - 1, history, 1, ctx, p,
                            same ? alpha : -beta, same ? beta : -alpha);
        int score = same ? raw : -raw;
        delete next;

        if(score > best_score){
            best_score       = score;
            result.best_move = action;
            result.score     = best_score;
            if(score > alpha) alpha = score;

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth,
                                    move_index + 1, total_moves});
            }
        }
        move_index++;
    }

    result.nodes    = ctx.nodes;
    result.seldepth = ctx.seldepth;
    result.time_ms  = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                          std::chrono::steady_clock::now() - g_search_start).count();
    return result;
}


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval",       "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial",   "false"},
        {"UseAlphaBeta",    "true"},
        {"UsePVS",          "true"},
        {"UseQuiescence",   "true"},
        {"QuiescenceDepth", "4"},
        {"UseLMR",          "true"},
        {"UseKillers",      "true"},
        {"UseHistory",      "true"},
        {"UseNullMove",     "true"},
        {"LMRMinDepth",     "3"},
        {"LMRFullMoves",    "4"},
        {"NullMoveR",       "3"},
        {"UseTT",           "true"},
        {"UseFutility",     "true"},
        {"FutilityMargin",  "150"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval",       ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial",   ParamDef::CHECK, "true"},
        {"UseAlphaBeta",    ParamDef::CHECK, "true"},
        {"UsePVS",          ParamDef::CHECK, "true"},
        {"UseQuiescence",   ParamDef::CHECK, "true"},
        {"QuiescenceDepth", ParamDef::SPIN,  "4",   0,   16},
        {"UseLMR",          ParamDef::CHECK, "true"},
        {"UseKillers",      ParamDef::CHECK, "true"},
        {"UseHistory",      ParamDef::CHECK, "true"},
        {"UseNullMove",     ParamDef::CHECK, "true"},
        {"LMRMinDepth",     ParamDef::SPIN,  "3",   1,   10},
        {"LMRFullMoves",    ParamDef::SPIN,  "4",   1,   20},
        {"NullMoveR",       ParamDef::SPIN,  "3",   1,   4},
        {"UseTT",           ParamDef::CHECK, "true"},
        {"UseFutility",     ParamDef::CHECK, "true"},
        {"FutilityMargin",  ParamDef::SPIN,  "150", 0,   500},
    };
}
