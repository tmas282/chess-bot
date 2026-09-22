#include "chess.h"
#include <string.h>
#include <stdlib.h>

/* ========================================================================== */
/* PRECOMPUTED TABLES                                                         */
/* ========================================================================== */

static Bitboard pawn_att_tbl[2][64];
static Bitboard knight_att_tbl[64];
static Bitboard king_att_tbl[64];
static Bitboard ray_n[64], ray_s[64], ray_e[64], ray_w[64];
static Bitboard ray_ne[64], ray_nw[64], ray_se[64], ray_sw[64];
static bool tables_init = false;

/* BERF <-> standard coordinate helpers */
static inline int sq_to_stdf(Square sq) { return 7 - (sq & 7); }
static inline int sq_to_stdr(Square sq) { return 8 - (sq >> 3); }
static inline Square std_to_sq(int f, int r) {
    return (Square)((8 - r) * 8 + (7 - f));
}

void init_tables(void) {
    if (tables_init) return;

    for (int sq = 0; sq < 64; sq++) {
        int f = sq_to_stdf((Square)sq);
        int r = sq_to_stdr((Square)sq);

        /* Pawn attacks */
        for (int c = 0; c < 2; c++) pawn_att_tbl[c][sq] = 0;
        /* White pawn: captures NE (f+1,r+1) [right], NW (f-1,r+1) [left] */
        if (f < 7) pawn_att_tbl[WHITE][sq] |= sq_bb(std_to_sq(f + 1, r + 1));
        if (f > 0) pawn_att_tbl[WHITE][sq] |= sq_bb(std_to_sq(f - 1, r + 1));
        /* Black pawn: captures SE (f+1,r-1) [left from black], SW (f-1,r-1) [right from black] */
        if (f < 7) pawn_att_tbl[BLACK][sq] |= sq_bb(std_to_sq(f + 1, r - 1));
        if (f > 0) pawn_att_tbl[BLACK][sq] |= sq_bb(std_to_sq(f - 1, r - 1));

        /* Knight attacks */
        int knight_df[] = { 1, 2, 2, 1, -1, -2, -2, -1 };
        int knight_dr[] = { 2, 1, -1, -2, -2, -1, 1, 2 };
        for (int i = 0; i < 8; i++) {
            int nf = f + knight_df[i], nr = r + knight_dr[i];
            if (nf >= 0 && nf < 8 && nr >= 1 && nr <= 8)
                knight_att_tbl[sq] |= sq_bb(std_to_sq(nf, nr));
        }

        /* King attacks */
        for (int df = -1; df <= 1; df++) {
            for (int dr = -1; dr <= 1; dr++) {
                if (df == 0 && dr == 0) continue;
                int nf = f + df, nr = r + dr;
                if (nf >= 0 && nf < 8 && nr >= 1 && nr <= 8)
                    king_att_tbl[sq] |= sq_bb(std_to_sq(nf, nr));
            }
        }

        /* Ray tables */
        int bf = sq & 7, br = sq >> 3;

        /* North: -8 per step, same file */
        for (int s = sq - 8; s >= 0; s -= 8)
            if ((s & 7) == bf) ray_n[sq] |= sq_bb((Square)s);

        /* South: +8 per step, same file */
        for (int s = sq + 8; s < 64; s += 8)
            if ((s & 7) == bf) ray_s[sq] |= sq_bb((Square)s);

        /* East: -1 per step, same rank (decreasing file = toward H) */
        for (int s = sq - 1; s >= 0 && (s >> 3) == br; s--)
            ray_e[sq] |= sq_bb((Square)s);

        /* West: +1 per step, same rank (increasing file = toward A) */
        for (int s = sq + 1; s < 64 && (s >> 3) == br; s++)
            ray_w[sq] |= sq_bb((Square)s);

        /* NE: -9, file decreases, rank decreases */
        for (int s = sq - 9; s >= 0 && (s & 7) < (sq & 7) && (s >> 3) < (sq >> 3); s -= 9)
            ray_ne[sq] |= sq_bb((Square)s);

        /* NW: -7, file increases, rank decreases */
        for (int s = sq - 7; s >= 0 && (s & 7) > (sq & 7) && (s >> 3) < (sq >> 3); s -= 7)
            ray_nw[sq] |= sq_bb((Square)s);

        /* SE: +7, file decreases, rank increases */
        for (int s = sq + 7; s < 64 && (s & 7) < (sq & 7) && (s >> 3) > (sq >> 3); s += 7)
            ray_se[sq] |= sq_bb((Square)s);

        /* SW: +9, file increases, rank increases */
        for (int s = sq + 9; s < 64 && (s & 7) > (sq & 7) && (s >> 3) > (sq >> 3); s += 9)
            ray_sw[sq] |= sq_bb((Square)s);
    }

    tables_init = true;
}

/* ========================================================================== */
/* BOARD CREATION                                                             */
/* ========================================================================== */

Board board_default(void) {
    Board b;
    memset(&b, 0, sizeof(b));
    b.pieces[PIECE_PAWN][WHITE]   = PAWN_WHITE_INITIAL;
    b.pieces[PIECE_PAWN][BLACK]   = PAWN_BLACK_INITIAL;
    b.pieces[PIECE_ROOK][WHITE]   = ROOK_WHITE_INITIAL;
    b.pieces[PIECE_ROOK][BLACK]   = ROOK_BLACK_INITIAL;
    b.pieces[PIECE_KNIGHT][WHITE] = KNIGHT_WHITE_INITIAL;
    b.pieces[PIECE_KNIGHT][BLACK] = KNIGHT_BLACK_INITIAL;
    b.pieces[PIECE_BISHOP][WHITE] = BISHOP_WHITE_INITIAL;
    b.pieces[PIECE_BISHOP][BLACK] = BISHOP_BLACK_INITIAL;
    b.pieces[PIECE_QUEEN][WHITE]  = QUEEN_WHITE_INITIAL;
    b.pieces[PIECE_QUEEN][BLACK]  = QUEEN_BLACK_INITIAL;
    b.pieces[PIECE_KING][WHITE]   = KING_WHITE_INITIAL;
    b.pieces[PIECE_KING][BLACK]   = KING_BLACK_INITIAL;
    b.side       = WHITE;
    b.ep         = (Square)-1;
    b.castling   = CASTLE_K | CASTLE_Q | CASTLE_k | CASTLE_q;
    b.halfmove   = 0;
    b.fullmove   = 1;
    return b;
}

/* ========================================================================== */
/* FEN PARSING / GENERATION                                                   */
/* ========================================================================== */

static int char_to_piece(char c) {
    switch (c) {
        case 'P': case 'p': return PIECE_PAWN;
        case 'N': case 'n': return PIECE_KNIGHT;
        case 'B': case 'b': return PIECE_BISHOP;
        case 'R': case 'r': return PIECE_ROOK;
        case 'Q': case 'q': return PIECE_QUEEN;
        case 'K': case 'k': return PIECE_KING;
        default:  return PIECE_NONE;
    }
}

static char *fen_skip_spaces(char *p) {
    while (*p == ' ') p++;
    return p;
}

void board_from_fen(Board *b, const char *fen) {
    memset(b, 0, sizeof(*b));
    b->ep = (Square)-1;

    char buf[256];
    strncpy(buf, fen, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;

    char *p = buf;

    /* Parse piece placement — read until first space */
    char *pieces = p;
    while (*p && *p != ' ') p++;
    if (*p) *p++ = '\0';
    p = fen_skip_spaces(p);

    int sr = 8, sc = 0;
    for (int i = 0; pieces[i]; i++) {
        char c = pieces[i];
        if (c == '/') { sr--; sc = 0; }
        else if (c >= '1' && c <= '8') { sc += c - '0'; }
        else {
            int color = (c >= 'A' && c <= 'Z') ? WHITE : BLACK;
            int pt = char_to_piece(c);
            if (pt >= 0) {
                Square sq = std_to_sq(sc, sr);
                b->pieces[pt][color] |= sq_bb(sq);
            }
            sc++;
        }
    }

    /* Side to move */
    if (*p) {
        b->side = (*p == 'w') ? WHITE : BLACK;
        p++;
        p = fen_skip_spaces(p);
    }

    /* Castling */
    if (*p) {
        if (*p != '-') {
            while (*p && *p != ' ') {
                switch (*p) {
                    case 'K': b->castling |= CASTLE_K; break;
                    case 'Q': b->castling |= CASTLE_Q; break;
                    case 'k': b->castling |= CASTLE_k; break;
                    case 'q': b->castling |= CASTLE_q; break;
                }
                p++;
            }
        } else {
            p++; /* skip '-' */
        }
        p = fen_skip_spaces(p);
    }

    /* En passant */
    if (*p) {
        if (*p != '-') {
            int ef = p[0] - 'a';
            int er = p[1] - '0';
            if (ef >= 0 && ef < 8 && er >= 1 && er <= 8)
                b->ep = std_to_sq(ef, er);
            p += 2;
        } else {
            p++;
        }
        p = fen_skip_spaces(p);
    }

    /* Halfmove clock */
    if (*p) {
        b->halfmove = (uint8_t)strtol(p, &p, 10);
        p = fen_skip_spaces(p);
    }

    /* Fullmove number */
    if (*p) {
        b->fullmove = (uint16_t)strtol(p, NULL, 10);
    }
}

static const char *piece_char = "PNBRQKpnbrqk";

void board_to_fen(const Board *b, char *fen) {
    int idx = 0;
    for (int sr = 8; sr >= 1; sr--) {
        int empty = 0;
        for (int sf = 0; sf < 8; sf++) {
            Square sq = std_to_sq(sf, sr);
            int pt;
            int color;
            bool found = false;
            for (pt = 0; pt < 6 && !found; pt++) {
                for (color = 0; color < 2 && !found; color++) {
                    if (b->pieces[pt][color] & sq_bb(sq)) {
                        found = true;
                    }
                }
            }
            pt--; color--;

            if (found) {
                if (empty) { fen[idx++] = '0' + empty; empty = 0; }
                int pi = color == WHITE ? pt : pt + 6;
                fen[idx++] = piece_char[pi];
            } else {
                empty++;
            }
        }
        if (empty) fen[idx++] = '0' + empty;
        if (sr > 1) fen[idx++] = '/';
    }
    fen[idx++] = ' ';
    fen[idx++] = b->side == WHITE ? 'w' : 'b';
    fen[idx++] = ' ';
    if (b->castling == 0) {
        fen[idx++] = '-';
    } else {
        if (b->castling & CASTLE_K) fen[idx++] = 'K';
        if (b->castling & CASTLE_Q) fen[idx++] = 'Q';
        if (b->castling & CASTLE_k) fen[idx++] = 'k';
        if (b->castling & CASTLE_q) fen[idx++] = 'q';
    }
    fen[idx++] = ' ';
    if (b->ep == (Square)-1) {
        fen[idx++] = '-';
    } else {
        int ef = sq_to_stdf(b->ep);
        int er = sq_to_stdr(b->ep);
        fen[idx++] = 'a' + ef;
        fen[idx++] = '0' + er;
    }
    fen[idx++] = ' ';
    int hm = b->halfmove;
    if (hm >= 100) { fen[idx++] = '0' + hm / 100; hm %= 100; }
    if (hm >= 10)  { fen[idx++] = '0' + hm / 10;   hm %= 10; }
    fen[idx++] = '0' + hm;
    fen[idx++] = ' ';
    int fm = b->fullmove;
    if (fm >= 100) { fen[idx++] = '0' + fm / 100; fm %= 100; }
    if (fm >= 10)  { fen[idx++] = '0' + fm / 10;   fm %= 10; }
    fen[idx++] = '0' + fm;
    fen[idx] = 0;
}

/* ========================================================================== */
/* PRINT                                                                      */
/* ========================================================================== */

void print_bitboard(Bitboard bb) {
    for (int r = 0; r < 8; r++) {
        for (int f = 0; f < 8; f++) {
            Square sq = std_to_sq(f, 8 - r);
            putchar(bb & sq_bb(sq) ? '1' : '0');
        }
        putchar('\n');
    }
}

void print_board(const Board *b) {
    const char *pieces = " PNBRQK pnbrqk";

    printf("  +-----------------+\n");
    for (int br = 0; br < 8; br++) {
        int sr = 8 - br;
        printf("%d | ", sr);
        for (int sf = 0; sf < 8; sf++) {
            Square sq = std_to_sq(sf, sr);
            int idx = 0;
            for (int pt = 0; pt < 6 && idx == 0; pt++) {
                if (b->pieces[pt][WHITE] & sq_bb(sq)) { idx = pt + 1; break; }
                if (b->pieces[pt][BLACK] & sq_bb(sq)) { idx = pt + 8; break; }
            }
            printf("%c ", idx ? pieces[idx] : '.');
        }
        printf("|\n");
    }
    printf("  +-----------------+\n");
    printf("    a b c d e f g h\n\n");

    printf("Side: %s  Castling: ", b->side == WHITE ? "White" : "Black");
    if (!b->castling) printf("-");
    if (b->castling & CASTLE_K) printf("K");
    if (b->castling & CASTLE_Q) printf("Q");
    if (b->castling & CASTLE_k) printf("k");
    if (b->castling & CASTLE_q) printf("q");
    printf("  EP: ");
    if (b->ep == (Square)-1) printf("-");
    else { int ef = sq_to_stdf(b->ep), er = sq_to_stdr(b->ep); printf("%c%d", 'a'+ef, er); }
    printf("  HM: %d  FM: %d\n", b->halfmove, b->fullmove);
}

/* ========================================================================== */
/* ATTACK GENERATORS                                                          */
/* ========================================================================== */

Bitboard pawn_attacks(Square sq, int color) {
    return pawn_att_tbl[color][sq];
}

Bitboard knight_attacks(Square sq) {
    return knight_att_tbl[sq];
}

Bitboard king_attacks(Square sq) {
    return king_att_tbl[sq];
}

Bitboard rook_attacks(Square sq, Bitboard occ) {
    Bitboard att = 0;

    /* North: step < 0 → MSB blocker */
    Bitboard rn = ray_n[sq];
    Bitboard bn = rn & occ;
    if (bn) rn ^= ray_n[msb_idx(bn)];
    att |= rn;

    /* South: step > 0 → LSB blocker */
    Bitboard rs = ray_s[sq];
    Bitboard bs = rs & occ;
    if (bs) rs ^= ray_s[lsb_idx(bs)];
    att |= rs;

    /* East: step < 0 → MSB blocker */
    Bitboard re = ray_e[sq];
    Bitboard be = re & occ;
    if (be) re ^= ray_e[msb_idx(be)];
    att |= re;

    /* West: step > 0 → LSB blocker */
    Bitboard rw = ray_w[sq];
    Bitboard bw = rw & occ;
    if (bw) rw ^= ray_w[lsb_idx(bw)];
    att |= rw;

    return att;
}

Bitboard bishop_attacks(Square sq, Bitboard occ) {
    Bitboard att = 0;

    Bitboard rne = ray_ne[sq];  Bitboard bne = rne & occ;
    if (bne) rne ^= ray_ne[msb_idx(bne)];
    att |= rne;

    Bitboard rnw = ray_nw[sq];  Bitboard bnw = rnw & occ;
    if (bnw) rnw ^= ray_nw[msb_idx(bnw)];
    att |= rnw;

    Bitboard rse = ray_se[sq];  Bitboard bse = rse & occ;
    if (bse) rse ^= ray_se[lsb_idx(bse)];
    att |= rse;

    Bitboard rsw = ray_sw[sq];  Bitboard bsw = rsw & occ;
    if (bsw) rsw ^= ray_sw[lsb_idx(bsw)];
    att |= rsw;

    return att;
}

Bitboard queen_attacks(Square sq, Bitboard occ) {
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ);
}

Bitboard piece_attacks(int piece, Square sq, Bitboard occ, int color) {
    switch (piece) {
        case PIECE_PAWN:   return pawn_attacks(sq, color);
        case PIECE_KNIGHT: return knight_attacks(sq);
        case PIECE_BISHOP: return bishop_attacks(sq, occ);
        case PIECE_ROOK:   return rook_attacks(sq, occ);
        case PIECE_QUEEN:  return queen_attacks(sq, occ);
        case PIECE_KING:   return king_attacks(sq);
        default:           return 0;
    }
}

/* ========================================================================== */
/* CHECK / ATTACK QUERIES                                                     */
/* ========================================================================== */

bool is_sq_attacked(const Board *b, Square sq, int by_color) {
    Bitboard occ = board_occ(b);
    int opp = by_color ^ 1;

    if (pawn_att_tbl[opp][sq] & b->pieces[PIECE_PAWN][by_color]) return true;
    if (knight_att_tbl[sq] & b->pieces[PIECE_KNIGHT][by_color]) return true;
    if (king_att_tbl[sq] & b->pieces[PIECE_KING][by_color]) return true;
    if (rook_attacks(sq, occ) & (b->pieces[PIECE_ROOK][by_color] |
                                  b->pieces[PIECE_QUEEN][by_color])) return true;
    if (bishop_attacks(sq, occ) & (b->pieces[PIECE_BISHOP][by_color] |
                                    b->pieces[PIECE_QUEEN][by_color])) return true;
    return false;
}

bool is_check(const Board *b) {
    Square ksq = king_sq(b, b->side);
    return is_sq_attacked(b, ksq, b->side ^ 1);
}

/* ========================================================================== */
/* MOVE GENERATION — PSEUDO-LEGAL                                            */
/* ========================================================================== */

static int gen_pawn_moves(const Board *b, Move *moves, int color,
                          Bitboard occ, Bitboard enemy, Bitboard enemy_king) {
    int cnt = 0;
    Bitboard pawns = b->pieces[PIECE_PAWN][color];
    int push_dir = (color == WHITE) ? -8 : 8;
    int double_rank = (color == WHITE) ? 6 : 1;
    int promo_rank = (color == WHITE) ? 0 : 7;

    while (pawns) {
        Square from = lsb_idx(pawns);
        pawns &= pawns - 1;

        /* Single push */
        Square to = (Square)(from + push_dir);
        if (to >= 0 && to < 64 && !(occ & sq_bb(to))) {
            if ((int)(to >> 3) == promo_rank) {
                /* Promotion */
                for (int pt = PIECE_KNIGHT; pt <= PIECE_QUEEN; pt++) {
                    int flag = (pt == PIECE_QUEEN) ? MOVE_QUEEN_PROMO :
                               (pt == PIECE_ROOK)  ? MOVE_ROOK_PROMO :
                               (pt == PIECE_BISHOP) ? MOVE_BISHOP_PROMO :
                                                       MOVE_KNIGHT_PROMO;
                    moves[cnt++] = MOVE_ENCODE(from, to, flag);
                }
            } else {
                moves[cnt++] = MOVE_ENCODE(from, to, MOVE_QUIET);
                /* Double push */
                if ((int)(from >> 3) == double_rank) {
                    Square to2 = (Square)(from + 2 * push_dir);
                    if (to2 >= 0 && to2 < 64 && !(occ & sq_bb(to2)))
                        moves[cnt++] = MOVE_ENCODE(from, to2, MOVE_DOUBLE_PAWN);
                }
            }
        }

        /* Captures */
        Bitboard caps = pawn_att_tbl[color][from] & enemy & ~enemy_king;
        while (caps) {
            Square capt = lsb_idx(caps);
            caps &= caps - 1;
            if ((int)(capt >> 3) == promo_rank) {
                for (int pt = PIECE_KNIGHT; pt <= PIECE_QUEEN; pt++) {
                    int flag = (pt == PIECE_QUEEN) ? MOVE_QUEEN_PROMO_CAP :
                               (pt == PIECE_ROOK)  ? MOVE_ROOK_PROMO_CAP :
                               (pt == PIECE_BISHOP) ? MOVE_BISHOP_PROMO_CAP :
                                                      MOVE_KNIGHT_PROMO_CAP;
                    moves[cnt++] = MOVE_ENCODE(from, capt, flag);
                }
            } else {
                moves[cnt++] = MOVE_ENCODE(from, capt, MOVE_CAPTURE);
            }
        }

        /* En passant */
        if (b->ep != (Square)-1) {
            Bitboard ep_caps = pawn_att_tbl[color][from] & sq_bb(b->ep);
            if (ep_caps) {
                moves[cnt++] = MOVE_ENCODE(from, b->ep, MOVE_EN_PASSANT);
            }
        }
    }
    return cnt;
}

static int gen_knight_moves(const Board *b, Move *moves, int color,
                            Bitboard occ_us, Bitboard enemy, Bitboard enemy_king) {
    int cnt = 0;
    Bitboard knights = b->pieces[PIECE_KNIGHT][color];
    while (knights) {
        Square from = lsb_idx(knights);
        knights &= knights - 1;
        Bitboard targets = knight_att_tbl[from] & ~occ_us & ~enemy_king;
        while (targets) {
            Square to = lsb_idx(targets);
            targets &= targets - 1;
            int flag = (enemy & sq_bb(to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[cnt++] = MOVE_ENCODE(from, to, flag);
        }
    }
    return cnt;
}

static int gen_bishop_moves(const Board *b, Move *moves, int color,
                            Bitboard occ_us, Bitboard enemy, Bitboard occ_all,
                            Bitboard enemy_king) {
    int cnt = 0;
    Bitboard bishops = b->pieces[PIECE_BISHOP][color];
    while (bishops) {
        Square from = lsb_idx(bishops);
        bishops &= bishops - 1;
        Bitboard targets = bishop_attacks(from, occ_all) & ~occ_us & ~enemy_king;
        while (targets) {
            Square to = lsb_idx(targets);
            targets &= targets - 1;
            int flag = (enemy & sq_bb(to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[cnt++] = MOVE_ENCODE(from, to, flag);
        }
    }
    return cnt;
}

static int gen_rook_moves(const Board *b, Move *moves, int color,
                          Bitboard occ_us, Bitboard enemy, Bitboard occ_all,
                          Bitboard enemy_king) {
    int cnt = 0;
    Bitboard rooks = b->pieces[PIECE_ROOK][color];
    while (rooks) {
        Square from = lsb_idx(rooks);
        rooks &= rooks - 1;
        Bitboard targets = rook_attacks(from, occ_all) & ~occ_us & ~enemy_king;
        while (targets) {
            Square to = lsb_idx(targets);
            targets &= targets - 1;
            int flag = (enemy & sq_bb(to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[cnt++] = MOVE_ENCODE(from, to, flag);
        }
    }
    return cnt;
}

static int gen_queen_moves(const Board *b, Move *moves, int color,
                           Bitboard occ_us, Bitboard enemy, Bitboard occ_all,
                           Bitboard enemy_king) {
    int cnt = 0;
    Bitboard queens = b->pieces[PIECE_QUEEN][color];
    while (queens) {
        Square from = lsb_idx(queens);
        queens &= queens - 1;
        Bitboard targets = queen_attacks(from, occ_all) & ~occ_us & ~enemy_king;
        while (targets) {
            Square to = lsb_idx(targets);
            targets &= targets - 1;
            int flag = (enemy & sq_bb(to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[cnt++] = MOVE_ENCODE(from, to, flag);
        }
    }
    return cnt;
}

static int gen_king_moves(const Board *b, Move *moves, int color,
                          Bitboard occ_us, Bitboard enemy, Bitboard occ_all,
                          Bitboard enemy_king) {
    int cnt = 0;
    Square from = king_sq(b, color);
    Bitboard targets = king_att_tbl[from] & ~occ_us & ~enemy_king;
    while (targets) {
        Square to = lsb_idx(targets);
        targets &= targets - 1;
        int flag = (enemy & sq_bb(to)) ? MOVE_CAPTURE : MOVE_QUIET;
        moves[cnt++] = MOVE_ENCODE(from, to, flag);
    }

    /* Cannot castle if in check */
    if (is_sq_attacked(b, from, color ^ 1)) return cnt;

    Bitboard empty = ~occ_all;

    /* Kingside (O-O) */
    if (color == WHITE) {
        if ((b->castling & CASTLE_K) && (b->pieces[PIECE_ROOK][WHITE] & sq_bb(H1)) &&
            (empty & sq_bb(F1)) && (empty & sq_bb(G1)) &&
            !is_sq_attacked(b, F1, BLACK) && !is_sq_attacked(b, G1, BLACK))
            moves[cnt++] = MOVE_ENCODE(from, G1, MOVE_KING_CASTLE);
    } else {
        if ((b->castling & CASTLE_k) && (b->pieces[PIECE_ROOK][BLACK] & sq_bb(H8)) &&
            (empty & sq_bb(F8)) && (empty & sq_bb(G8)) &&
            !is_sq_attacked(b, F8, WHITE) && !is_sq_attacked(b, G8, WHITE))
            moves[cnt++] = MOVE_ENCODE(from, G8, MOVE_KING_CASTLE);
    }

    /* Queenside (O-O-O) */
    if (color == WHITE) {
        if ((b->castling & CASTLE_Q) && (b->pieces[PIECE_ROOK][WHITE] & sq_bb(A1)) &&
            (empty & sq_bb(B1)) && (empty & sq_bb(C1)) && (empty & sq_bb(D1)) &&
            !is_sq_attacked(b, C1, BLACK) && !is_sq_attacked(b, D1, BLACK))
            moves[cnt++] = MOVE_ENCODE(from, C1, MOVE_QUEEN_CASTLE);
    } else {
        if ((b->castling & CASTLE_q) && (b->pieces[PIECE_ROOK][BLACK] & sq_bb(A8)) &&
            (empty & sq_bb(B8)) && (empty & sq_bb(C8)) && (empty & sq_bb(D8)) &&
            !is_sq_attacked(b, C8, WHITE) && !is_sq_attacked(b, D8, WHITE))
            moves[cnt++] = MOVE_ENCODE(from, C8, MOVE_QUEEN_CASTLE);
    }

    return cnt;
}

int gen_pseudo_moves(const Board *b, Move *moves) {
    int cnt = 0;
    int color = b->side;
    int enemy = color ^ 1;

    Bitboard occ_us = board_occ_color(b, color);
    Bitboard enemy_occ = board_occ_color(b, enemy);
    Bitboard occ_all = occ_us | enemy_occ;
    Bitboard enemy_king = b->pieces[PIECE_KING][enemy];

    cnt += gen_pawn_moves(b, moves + cnt, color, occ_all, enemy_occ, enemy_king);
    cnt += gen_knight_moves(b, moves + cnt, color, occ_us, enemy_occ, enemy_king);
    cnt += gen_bishop_moves(b, moves + cnt, color, occ_us, enemy_occ, occ_all, enemy_king);
    cnt += gen_rook_moves(b, moves + cnt, color, occ_us, enemy_occ, occ_all, enemy_king);
    cnt += gen_queen_moves(b, moves + cnt, color, occ_us, enemy_occ, occ_all, enemy_king);
    cnt += gen_king_moves(b, moves + cnt, color, occ_us, enemy_occ, occ_all, enemy_king);

    return cnt;
}

/* ========================================================================== */
/* LEGAL MOVE FILTERING                                                       */
/* ========================================================================== */

bool is_legal_move(const Board *b, Move move) {
    int color = b->side;

    /* Quick check: king cannot be captured (not needed for pseudo-legal) */

    /* Make a copy and apply the move */
    Board copy = *b;
    apply_move(&copy, move);

    /* After move, own king must not be in check */
    Square ksq = king_sq(&copy, color);
    if (is_sq_attacked(&copy, ksq, color ^ 1))
        return false;

    return true;
}

int gen_legal_moves(const Board *b, Move *moves) {
    Move pseudo[256];
    int np = gen_pseudo_moves(b, pseudo);
    int cnt = 0;

    for (int i = 0; i < np; i++) {
        /* For speed, handle a few common cases without full board copy:
         * - If the king moved, check the destination square */
        if (is_legal_move(b, pseudo[i]))
            moves[cnt++] = pseudo[i];
    }

    return cnt;
}

/* ========================================================================== */
/* APPLY MOVE                                                                 */
/* ========================================================================== */

void apply_move(Board *b, Move move) {
    Square from = MOVE_GET_FROM(move);
    Square to = MOVE_GET_TO(move);
    int flag = MOVE_GET_FLAG(move);
    int color = b->side;
    int enemy = color ^ 1;

    Bitboard from_bb = sq_bb(from);
    Bitboard to_bb = sq_bb(to);

    b->ep = (Square)-1;

    /* Clear the from square for the moving piece */
    int moving_pt = piece_on(b, from);
    if (moving_pt < 0) return; /* Should not happen */

    b->pieces[moving_pt][color] &= ~from_bb;

    /* Handle special moves */
    switch (flag) {
        case MOVE_DOUBLE_PAWN: {
            b->pieces[moving_pt][color] |= to_bb;
            /* Set en passant target (intermediate square) */
            int ep_sq = (from + to) / 2;
            b->ep = (Square)ep_sq;
            break;
        }
        case MOVE_EN_PASSANT: {
            /* Remove captured pawn (it's on the rank behind the destination) */
            int cap_sq = (color == WHITE) ? to + 8 : to - 8;
            b->pieces[PIECE_PAWN][enemy] &= ~sq_bb((Square)cap_sq);
            b->pieces[moving_pt][color] |= to_bb;
            break;
        }
        case MOVE_KING_CASTLE: {
            b->pieces[PIECE_KING][color] |= to_bb;
            /* Move rook */
            int rsrc = (color == WHITE) ? H1 : H8;
            int rdst = (color == WHITE) ? F1 : F8;
            b->pieces[PIECE_ROOK][color] &= ~sq_bb(rsrc);
            b->pieces[PIECE_ROOK][color] |= sq_bb(rdst);
            /* Remove castling rights for this color */
            if (color == WHITE) b->castling &= ~(CASTLE_K | CASTLE_Q);
            else b->castling &= ~(CASTLE_k | CASTLE_q);
            break;
        }
        case MOVE_QUEEN_CASTLE: {
            b->pieces[PIECE_KING][color] |= to_bb;
            /* Move rook */
            int rsrc = (color == WHITE) ? A1 : A8;
            int rdst = (color == WHITE) ? D1 : D8;
            b->pieces[PIECE_ROOK][color] &= ~sq_bb(rsrc);
            b->pieces[PIECE_ROOK][color] |= sq_bb(rdst);
            if (color == WHITE) b->castling &= ~(CASTLE_K | CASTLE_Q);
            else b->castling &= ~(CASTLE_k | CASTLE_q);
            break;
        }
        case MOVE_KNIGHT_PROMO:
        case MOVE_BISHOP_PROMO:
        case MOVE_ROOK_PROMO:
        case MOVE_QUEEN_PROMO: {
            int prom_pt;
            if (flag == MOVE_KNIGHT_PROMO) prom_pt = PIECE_KNIGHT;
            else if (flag == MOVE_BISHOP_PROMO) prom_pt = PIECE_BISHOP;
            else if (flag == MOVE_ROOK_PROMO) prom_pt = PIECE_ROOK;
            else prom_pt = PIECE_QUEEN;
            b->pieces[PIECE_PAWN][color] &= ~to_bb; /* Remove pawn from destination */
            b->pieces[prom_pt][color] |= to_bb;
            break;
        }
        case MOVE_KNIGHT_PROMO_CAP:
        case MOVE_BISHOP_PROMO_CAP:
        case MOVE_ROOK_PROMO_CAP:
        case MOVE_QUEEN_PROMO_CAP: {
            int prom_pt;
            if (flag == MOVE_KNIGHT_PROMO_CAP) prom_pt = PIECE_KNIGHT;
            else if (flag == MOVE_BISHOP_PROMO_CAP) prom_pt = PIECE_BISHOP;
            else if (flag == MOVE_ROOK_PROMO_CAP) prom_pt = PIECE_ROOK;
            else prom_pt = PIECE_QUEEN;
            /* Remove captured piece */
            for (int pt = 0; pt < 6; pt++)
                b->pieces[pt][enemy] &= ~to_bb;
            b->pieces[PIECE_PAWN][color] &= ~to_bb;
            b->pieces[prom_pt][color] |= to_bb;
            break;
        }
        default: {
            /* Normal move or capture */
            if (flag == MOVE_CAPTURE) {
                /* Remove captured piece */
                for (int pt = 0; pt < 6; pt++)
                    b->pieces[pt][enemy] &= ~to_bb;
            }
            b->pieces[moving_pt][color] |= to_bb;

            /* Check for rook capture that removes castling rights */
            if (to_bb & sq_bb(H1)) b->castling &= ~CASTLE_K;
            if (to_bb & sq_bb(A1)) b->castling &= ~CASTLE_Q;
            if (to_bb & sq_bb(H8)) b->castling &= ~CASTLE_k;
            if (to_bb & sq_bb(A8)) b->castling &= ~CASTLE_q;
            break;
        }
    }

    /* Update castling rights if king or rook moved */
    if (moving_pt == PIECE_KING) {
        if (color == WHITE) b->castling &= ~(CASTLE_K | CASTLE_Q);
        else b->castling &= ~(CASTLE_k | CASTLE_q);
    }
    if (moving_pt == PIECE_ROOK) {
        if (from_bb & sq_bb(H1)) b->castling &= ~CASTLE_K;
        if (from_bb & sq_bb(A1)) b->castling &= ~CASTLE_Q;
        if (from_bb & sq_bb(H8)) b->castling &= ~CASTLE_k;
        if (from_bb & sq_bb(A8)) b->castling &= ~CASTLE_q;
    }

    /* Switch side */
    b->side = enemy;

    /* Update move clocks */
    if (moving_pt == PIECE_PAWN || flag == MOVE_CAPTURE ||
        flag == MOVE_EN_PASSANT || flag >= MOVE_KNIGHT_PROMO)
        b->halfmove = 0;
    else
        b->halfmove++;

    if (color == BLACK) b->fullmove++;
}

/* ========================================================================== */
/* CHECKMATE / STALEMATE                                                      */
/* ========================================================================== */

bool is_checkmate(const Board *b) {
    if (!is_check(b)) return false;
    Move moves[256];
    return gen_legal_moves(b, moves) == 0;
}

bool is_stalemate(const Board *b) {
    if (is_check(b)) return false;
    Move moves[256];
    return gen_legal_moves(b, moves) == 0;
}

/* ========================================================================== */
/* NN PLANE CONVERSION                                                        */
/* ========================================================================== */

void board_to_planes(const Board *b, uint8_t planes[14][8][8]) {
    for (int i = 0; i < 14; i++)
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                planes[i][r][c] = 0;

    /* Piece planes 0-11 */
    for (int sq = 0; sq < 64; sq++) {
        Bitboard bb = sq_bb((Square)sq);
        int nn_r = sq >> 3;
        int nn_c = 7 - (sq & 7);

        for (int pt = 0; pt < 6; pt++) {
            if (b->pieces[pt][WHITE] & bb)
                planes[pt][nn_r][nn_c] = 1;
            if (b->pieces[pt][BLACK] & bb)
                planes[pt + 6][nn_r][nn_c] = 1;
        }
    }

    /* Side to move plane 12 */
    if (b->side == WHITE)
        planes[12][0][0] = 1;

    /* Castling plane 13 */
    if (b->castling & CASTLE_K) planes[13][0][0] = 1;
    if (b->castling & CASTLE_Q) planes[13][0][1] = 1;
    if (b->castling & CASTLE_k) planes[13][0][2] = 1;
    if (b->castling & CASTLE_q) planes[13][0][3] = 1;
}

int move_to_uci(Move m, char *s) {
    Square from = MOVE_GET_FROM(m);
    Square to = MOVE_GET_TO(m);
    int flag = MOVE_GET_FLAG(m);
    int len = 4;

    s[0] = 'a' + file_of(from);
    s[1] = '0' + (8 - rank_of(from));
    s[2] = 'a' + file_of(to);
    s[3] = '0' + (8 - rank_of(to));

    if (flag == MOVE_KNIGHT_PROMO || flag == MOVE_KNIGHT_PROMO_CAP) {
        s[4] = 'n'; len = 5;
    } else if (flag == MOVE_BISHOP_PROMO || flag == MOVE_BISHOP_PROMO_CAP) {
        s[4] = 'b'; len = 5;
    } else if (flag == MOVE_ROOK_PROMO || flag == MOVE_ROOK_PROMO_CAP) {
        s[4] = 'r'; len = 5;
    } else if (flag == MOVE_QUEEN_PROMO || flag == MOVE_QUEEN_PROMO_CAP) {
        s[4] = 'q'; len = 5;
    }
    s[len] = 0;
    return len;
}

int uci_parse_move(const Board *b, const char *s, Move *m) {
    if (s[0] < 'a' || s[0] > 'h' || s[1] < '1' || s[1] > '8' ||
        s[2] < 'a' || s[2] > 'h' || s[3] < '1' || s[3] > '8')
        return 0;

    int ffrom = s[0] - 'a';
    int rfrom = s[1] - '0';
    int fto   = s[2] - 'a';
    int rto   = s[3] - '0';

    Square from = std_to_sq(ffrom, rfrom);
    Square to   = std_to_sq(fto, rto);
    int flag    = MOVE_QUIET;

    if (b) {
        int pt = piece_on(b, from);
        int captured = piece_on(b, to);

        if (pt == PIECE_PAWN) {
            int promo_rank = (b->side == WHITE) ? 0 : 7;
            if ((int)(to >> 3) == promo_rank) {
                char prom = s[4] ? s[4] : 'q';
                int is_cap = (captured >= 0 || (int)b->ep == (int)to);
                switch (prom) {
                    case 'q': flag = is_cap ? MOVE_QUEEN_PROMO_CAP : MOVE_QUEEN_PROMO; break;
                    case 'r': flag = is_cap ? MOVE_ROOK_PROMO_CAP : MOVE_ROOK_PROMO;   break;
                    case 'b': flag = is_cap ? MOVE_BISHOP_PROMO_CAP : MOVE_BISHOP_PROMO; break;
                    case 'n': flag = is_cap ? MOVE_KNIGHT_PROMO_CAP : MOVE_KNIGHT_PROMO; break;
                }
                *m = MOVE_ENCODE(from, to, flag);
                return 1;
            }

            int push_dir = (b->side == WHITE) ? -8 : 8;
            int double_from_rank = (b->side == WHITE) ? 6 : 1;
            if ((int)(from >> 3) == double_from_rank && (int)(to - from) == 2 * push_dir)
                flag = MOVE_DOUBLE_PAWN;
            else if (captured >= 0)
                flag = MOVE_CAPTURE;
            else if ((int)b->ep == (int)to)
                flag = MOVE_EN_PASSANT;
        } else if (pt == PIECE_KING) {
            int kside_to = (b->side == WHITE) ? (int)G1 : (int)G8;
            int qside_to = (b->side == WHITE) ? (int)C1 : (int)C8;
            int diff = (int)to - (int)from;
            if ((int)to == kside_to && (diff == 2 || diff == -2))
                flag = MOVE_KING_CASTLE;
            else if ((int)to == qside_to && (diff == 2 || diff == -2))
                flag = MOVE_QUEEN_CASTLE;
            else if (captured >= 0)
                flag = MOVE_CAPTURE;
        } else if (captured >= 0) {
            flag = MOVE_CAPTURE;
        }
    } else {
        if (ffrom == fto && abs(rfrom - rto) == 2)
            flag = MOVE_DOUBLE_PAWN;
        else if (s[0] == 'e' && s[2] == 'g' && (s[1] == '1' || s[1] == '8'))
            flag = MOVE_KING_CASTLE;
        else if (s[0] == 'e' && s[2] == 'c' && (s[1] == '1' || s[1] == '8'))
            flag = MOVE_QUEEN_CASTLE;

        if (s[4]) {
            switch (s[4]) {
                case 'n': flag = MOVE_KNIGHT_PROMO; break;
                case 'b': flag = MOVE_BISHOP_PROMO; break;
                case 'r': flag = MOVE_ROOK_PROMO;   break;
                case 'q': flag = MOVE_QUEEN_PROMO;  break;
            }
        }
    }

    *m = MOVE_ENCODE(from, to, flag);
    return 1;
}
