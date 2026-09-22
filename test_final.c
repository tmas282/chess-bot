#include "chess/chess.h"
#include "chess/minimax.h"
#include <stdio.h>
#include <string.h>

static int test_white_mate_in_1_preferred(void) {
    Board b;
    board_from_fen(&b, "k7/8/2K5/8/8/8/8/1Q6 w - - 0 1");
    Move best = minimax_init(&b);
    char buf[8];
    move_to_uci(best, buf);
    Board after = b;
    apply_move(&after, best);
    int result = is_checkmate(&after);
    printf("White mate-in-1: bestmove=%s is_mate=%d\n", buf, result);
    return result;
}

// Black queen a1 + black king b2 vs white king h1 — Qh1# is mate-in-1
static int test_black_mate_in_1_preferred(void) {
    Board b;
    board_from_fen(&b, "8/8/8/8/8/8/1k6/q6K b - - 0 1");
    print_board(&b);
    Move best = minimax_init(&b);
    char buf[8];
    move_to_uci(best, buf);
    Board after = b;
    apply_move(&after, best);
    int result = is_checkmate(&after);
    printf("Black mate-in-1: bestmove=%s is_mate=%d\n", buf, result);
    return result;
}

static int test_scholars_mate_defense(void) {
    // Scholar's mate position: BLACK is already in checkmate
    Board b;
    board_from_fen(&b, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    Move best = minimax_init(&b);
    int result = (best == 0);
    printf("Scholar's mate (already mate): bestmove=0000 -> %s\n", result ? "PASS" : "FAIL");
    return result;
}

static int test_scaled_scores(void) {
    Board b;
    board_from_fen(&b, "k7/8/2K5/8/8/8/8/1Q6 w - - 0 1");
    Move moves[256];
    short count = gen_legal_moves(&b, moves);

    float mate1_score = -999.0f;
    float other_win_score = -999.0f;
    for (short i = 0; i < count; i++) {
        Board child = clone_board_and_move(&b, &moves[i]);
        float score = minimax(&child, DEPTH, false);
        Board after = b;
        apply_move(&after, moves[i]);
        if (is_checkmate(&after)) {
            mate1_score = score;
        } else if (score > 0.0f) {
            other_win_score = score;
        }
    }
    int result = (mate1_score > other_win_score);
    printf("Score comparison: mate_in_1=%f > other_win=%f -> %s\n",
           mate1_score, other_win_score, result ? "PASS" : "FAIL");
    return result;
}

int main(void) {
    init_tables();
    int pass = 0, total = 4;
    pass += test_white_mate_in_1_preferred();
    pass += test_black_mate_in_1_preferred();
    pass += test_scholars_mate_defense();
    pass += test_scaled_scores();
    printf("\n=== Results: %d/%d passed ===\n", pass, total);
    return pass == total ? 0 : 1;
}
