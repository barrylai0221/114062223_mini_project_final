#pragma once
#include "search_types.hpp"
#include "game_history.hpp"

struct MMParams {
    bool use_kp_eval       = true;
    bool use_eval_mobility = true;
    bool report_partial    = false;
    bool use_ab            = true;   // alpha-beta pruning
    bool use_pvs           = true;   // principal variation search
    bool use_quiescence    = true;   // quiescence search at horizon
    int  q_depth           = 4;      // extra plies for quiescence
    bool use_lmr           = true;   // late move reduction
    bool use_killers       = true;   // killer heuristic
    bool use_history       = true;   // history heuristic
    bool use_null_move     = true;   // null move pruning
    int  lmr_min_depth     = 3;      // shallowest depth where LMR fires
    int  lmr_full_moves    = 4;      // moves searched at full depth before LMR
    int  null_move_r       = 3;      // null move depth reduction
    bool use_tt            = true;   // transposition table
    bool use_futility      = true;   // futility pruning at depth 1
    int  futility_margin   = 150;    // material margin for futility (eval units)

    static MMParams from_map(const ParamMap& m){
        MMParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval",       true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.report_partial    = param_bool(m, "ReportPartial",   true);
        p.use_ab            = param_bool(m, "UseAlphaBeta",    true);
        p.use_pvs           = param_bool(m, "UsePVS",          true);
        p.use_quiescence    = param_bool(m, "UseQuiescence",   true);
        p.q_depth           = param_int (m, "QuiescenceDepth", 4);
        p.use_lmr           = param_bool(m, "UseLMR",          true);
        p.use_killers       = param_bool(m, "UseKillers",      true);
        p.use_history       = param_bool(m, "UseHistory",      true);
        p.use_null_move     = param_bool(m, "UseNullMove",     true);
        p.lmr_min_depth     = param_int (m, "LMRMinDepth",     3);
        p.lmr_full_moves    = param_int (m, "LMRFullMoves",    4);
        p.null_move_r       = param_int (m, "NullMoveR",       3);
        p.use_tt            = param_bool(m, "UseTT",           true);
        p.use_futility      = param_bool(m, "UseFutility",     true);
        p.futility_margin   = param_int (m, "FutilityMargin",  150);
        return p;
    }
};

class MiniMax{
public:
    // was_null: true when the parent was a null move — prevents consecutive null moves
    static int eval_ctx(
        State *state,
        int depth,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const MMParams& p,
        int alpha    = M_MAX,
        int beta     = P_MAX,
        bool was_null = false
    );
    static SearchResult search(
        State *state,
        int depth,
        GameHistory& history,
        SearchContext& ctx
    );

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};
