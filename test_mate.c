#include "chess/chess.h"
#include "chess/minimax.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    init_tables();
    
    Board b;
    board_from_fen(&b, "k7/8/2K5/8/8/8/8/1Q6 w - - 0 1");
    
    // Test clone_board_and_move for Qb7
    Move move;
    uci_parse_move(&b, "b1b7", &move);
    
    Board cloned = clone_board_and_move(&b, &move);
    printf("After clone_board_and_move for Qb7:\n");
    print_board(&cloned);
    printf("side=%d\n", cloned.side);
    printf("is_check=%d is_checkmate=%d\n", is_check(&cloned), is_checkmate(&cloned));
    
    // Compare with direct apply_move
    Board direct = b;
    apply_move(&direct, move);
    printf("\nAfter direct apply_move for Qb7:\n");
    print_board(&direct);
    printf("side=%d\n", direct.side);
    printf("is_check=%d is_checkmate=%d\n", is_check(&direct), is_checkmate(&direct));
    
    // Also test all legal moves from minimax_init to see their scores
    printf("\n=== All moves score ===\n");
    Move moves[256];
    short count = gen_legal_moves(&b, moves);
    printf("Total legal moves: %d\n", count);
    for (short i = 0; i < count; i++) {
        Board b2 = clone_board_and_move(&b, &moves[i]);
        float score = minimax(&b2, DEPTH, false);
        char buf[8];
        move_to_uci(moves[i], buf);
        printf("  %s score=%f\n", buf, score);
    }
    
    // Also print the FEN from board_to_fen
    char fen[92];
    board_to_fen(&b, fen);
    printf("\nFEN: %s\n", fen);
    
    Board b_restored;
    board_from_fen(&b_restored, fen);
    printf("Restored board checkmate=%d\n", is_checkmate(&b_restored));
    
    return 0;
}
