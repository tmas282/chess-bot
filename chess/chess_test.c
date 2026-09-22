#include "chess.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

/* ========================================================================== */
/* Tests                                                                      */
/* ========================================================================== */

static int test_initial(void) {
    Board b = board_default();
    print_board(&b);

    Move moves[256];
    int n = gen_legal_moves(&b, moves);
    int ok = (n == 20 && !is_check(&b) && !is_checkmate(&b) && !is_stalemate(&b));
    printf("Initial position: %d legal moves (exp 20), check=%d, mate=%d, stalemate=%d \u2192 %s\n\n",
           n, is_check(&b), is_checkmate(&b), is_stalemate(&b), ok ? "PASS" : "FAIL");
    return ok;
}

static int test_scholars_mate(void) {
    char *fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4";
    Board b;
    board_from_fen(&b, fen);

    int check = is_check(&b);
    int mate = is_checkmate(&b);
    char fen_out[128];
    board_to_fen(&b, fen_out);
    int fen_ok = (strcmp(fen, fen_out) == 0);

    printf("Scholar's mate: check=%d, mate=%d, FEN=%s \u2192 %s\n\n",
           check, mate, fen_ok ? "match" : "mismatch",
           (check && mate && fen_ok) ? "PASS" : "FAIL");
    return check && mate && fen_ok;
}

static int test_en_passant(void) {
    char *fen = "4k3/8/8/8/Pp6/8/8/4K3 b - a3 0 1";
    Board b;
    board_from_fen(&b, fen);

    Move moves[256];
    int n = gen_legal_moves(&b, moves);
    int has_ep = 0;
    for (int i = 0; i < n; i++)
        if (MOVE_GET_FLAG(moves[i]) == MOVE_EN_PASSANT) has_ep = 1;

    printf("En passant: %d legal moves (expected ~7), EP capture=%d \u2192 %s\n\n",
           n, has_ep, (n >= 6 && has_ep) ? "PASS" : "FAIL");
    return (n >= 6 && has_ep);
}

static int test_castling(void) {
    char *fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    Board b;
    board_from_fen(&b, fen);

    Move moves[256];
    int n = gen_legal_moves(&b, moves);
    int has_oo = 0, has_ooo = 0;
    for (int i = 0; i < n; i++) {
        int f = MOVE_GET_FLAG(moves[i]);
        if (f == MOVE_KING_CASTLE) has_oo = 1;
        if (f == MOVE_QUEEN_CASTLE) has_ooo = 1;
    }

    printf("Castling: %d legal moves, O-O=%d, O-O-O=%d \u2192 %s\n\n",
           n, has_oo, has_ooo, (n > 20 && has_oo && has_ooo) ? "PASS" : "FAIL");
    return (n > 20 && has_oo && has_ooo);
}

static int test_nn_planes(void) {
    Board b = board_default();
    uint8_t planes[14][8][8] = {0};
    board_to_planes(&b, planes);

    int wp = 0, bp = 0;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++) {
            wp += planes[0][r][c];
            bp += planes[6][r][c];
        }

    int ok = (wp == 8 && bp == 8 && planes[12][0][0] == 1);
    printf("NN planes: white pawns=%d, black pawns=%d, side=%d \u2192 %s\n\n",
           wp, bp, planes[12][0][0], ok ? "PASS" : "FAIL");
    return ok;
}

static int test_apply_move(void) {
    Board b = board_default();
    Move moves[256];
    int n = gen_legal_moves(&b, moves);

    int found = 0;
    for (int i = 0; i < n; i++) {
        Square from = MOVE_GET_FROM(moves[i]);
        Square to = MOVE_GET_TO(moves[i]);
        if (from == E2 && to == E4 && MOVE_GET_FLAG(moves[i]) == MOVE_DOUBLE_PAWN) {
            apply_move(&b, moves[i]);
            found = 1;
            break;
        }
    }

    if (!found) { printf("Apply move e4: FAIL (move not found)\n\n"); return 0; }

    int ok = (b.side == BLACK &&
              piece_on(&b, E4) == PIECE_PAWN &&
              piece_on(&b, E2) == PIECE_NONE);
    printf("Apply 1.e4: side=%s, e4=pawn, e2=empty \u2192 %s\n\n",
           b.side == WHITE ? "white" : "black", ok ? "PASS" : "FAIL");
    return ok;
}

static void test_uci_parse_move(void) {
    char buf[8];
    Move m;
    int len;

    /* Quiet move */
    uci_parse_move(NULL, "e2e4", &m);
    assert(MOVE_GET_FROM(m) == E2);
    assert(MOVE_GET_TO(m) == E4);
    assert(MOVE_GET_FLAG(m) == MOVE_DOUBLE_PAWN);

    /* King-side castle white */
    uci_parse_move(NULL,"e1g1", &m);
    assert(MOVE_GET_FROM(m) == E1);
    assert(MOVE_GET_TO(m) == G1);
    assert(MOVE_GET_FLAG(m) == MOVE_KING_CASTLE);

    /* Queen-side castle black */
    uci_parse_move(NULL,"e8c8", &m);
    assert(MOVE_GET_FROM(m) == E8);
    assert(MOVE_GET_TO(m) == C8);
    assert(MOVE_GET_FLAG(m) == MOVE_QUEEN_CASTLE);

    /* Regular move */
    uci_parse_move(NULL,"g1f3", &m);
    assert(MOVE_GET_FROM(m) == G1);
    assert(MOVE_GET_TO(m) == F3);
    assert(MOVE_GET_FLAG(m) == MOVE_QUIET);

    /* Capture — no board, so quiet flag */
    uci_parse_move(NULL,"d5e6", &m);
    assert(MOVE_GET_FROM(m) == D5);
    assert(MOVE_GET_TO(m) == E6);
    assert(MOVE_GET_FLAG(m) == MOVE_QUIET);

    /* Promotion to queen */
    uci_parse_move(NULL,"e7e8q", &m);
    assert(MOVE_GET_FROM(m) == E7);
    assert(MOVE_GET_TO(m) == E8);
    assert(MOVE_GET_FLAG(m) == MOVE_QUEEN_PROMO);

    /* Promotion to knight */
    uci_parse_move(NULL,"a7a8n", &m);
    assert(MOVE_GET_FROM(m) == A7);
    assert(MOVE_GET_TO(m) == A8);
    assert(MOVE_GET_FLAG(m) == MOVE_KNIGHT_PROMO);

    /* Roundtrip: uci -> move -> uci */
    const char *inputs[] = {"e2e4","g1f3","d2d4","e1g1","e1c1","e8g8","e8c8","e7e8q","a7a8n"};
    for (int i = 0; i < 9; i++) {
        uci_parse_move(NULL,(char*)inputs[i], &m);
        len = move_to_uci(m, buf);
        buf[len] = 0;
        assert(strcmp(buf, inputs[i]) == 0);
    }

    printf("uci_to_move: all assertions passed\n\n");
}

int main(void) {
    init_tables();

    int pass = 0, total = 7;
    pass += test_initial();
    pass += test_scholars_mate();
    pass += test_en_passant();
    pass += test_castling();
    pass += test_nn_planes();
    pass += test_apply_move();
    test_uci_parse_move();
    pass++;

    printf("=== Results: %d/%d tests passed ===\n", pass, total);
    return pass == total ? 0 : 1;
}
