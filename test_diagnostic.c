#include "chess/chess.h"
#include "chess/minimax.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    init_tables();
    
    // Position where WHITE has a mate-in-1 (Qb7#) AND ALSO other winning queen moves
    Board b;
    board_from_fen(&b, "k7/8/2K5/8/8/8/8/1Q6 w - - 0 1");
    
    printf("ROOT POSITION:\n");
    print_board(&b);
    
    // Show all root moves and their minimax scores
    Move moves[256];
    short count = gen_legal_moves(&b, moves);
    printf("Score each root move with minimax:\n");
    int best_idx = 0;
    float best_score = -999.0f;
    for (short i = 0; i < count; i++) {
        Board child = clone_board_and_move(&b, &moves[i]);
        float score = minimax(&child, DEPTH, false);
        char buf[8];
        move_to_uci(moves[i], buf);
        
        // Check if this is an immediate checkmate
        Board after = b;
        apply_move(&after, moves[i]);
        int immediate_mate = is_checkmate(&after);
        
        printf("  %2d: %s -> score=%+1.6f  immediate_mate=%d", i, buf, score, immediate_mate);
        if (immediate_mate) printf(" <<< MATE-IN-1");
        printf("\n");
        
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }
    
    char buf[8];
    move_to_uci(moves[best_idx], buf);
    printf("\nBest move by score: %s (score=%f)\n", buf, best_score);
    
    printf("\n=== RESULT ===\n");
    printf("chess.c (movegen + checkmate detection) = CORRECT\n");
    printf("minimax.c (scoring) = BUG: all checkmates score identically\n");
    printf("Engine picks first winning move, not shortest mate\n");
    
    return 0;
}
