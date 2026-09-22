#ifndef CHESS_H
#define CHESS_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/* ========================================================================== */
/* TYPES                                                                      */
/* ========================================================================== */

typedef uint64_t Bitboard;

/* Colors */
enum { BLACK, WHITE };

/* Piece types in order matching NN plane indices */
enum {
    PIECE_PAWN,
    PIECE_KNIGHT,
    PIECE_BISHOP,
    PIECE_ROOK,
    PIECE_QUEEN,
    PIECE_KING,
    PIECE_NONE = -1
};

typedef enum {
    H8=0,  G8=1,  F8=2,  E8=3,  D8=4,  C8=5,  B8=6,  A8=7,
    H7=8,  G7=9,  F7=10, E7=11, D7=12, C7=13, B7=14, A7=15,
    H6=16, G6=17, F6=18, E6=19, D6=20, C6=21, B6=22, A6=23,
    H5=24, G5=25, F5=26, E5=27, D5=28, C5=29, B5=30, A5=31,
    H4=32, G4=33, F4=34, E4=35, D4=36, C4=37, B4=38, A4=39,
    H3=40, G3=41, F3=42, E3=43, D3=44, C3=45, B3=46, A3=47,
    H2=48, G2=49, F2=50, E2=51, D2=52, C2=53, B2=54, A2=55,
    H1=56, G1=57, F1=58, E1=59, D1=60, C1=61, B1=62, A1=63
} Square;

/* Move flags */
enum {
    MOVE_QUIET,
    MOVE_DOUBLE_PAWN,
    MOVE_KING_CASTLE,
    MOVE_QUEEN_CASTLE,
    MOVE_CAPTURE,
    MOVE_EN_PASSANT,
    MOVE_KNIGHT_PROMO,
    MOVE_BISHOP_PROMO,
    MOVE_ROOK_PROMO,
    MOVE_QUEEN_PROMO,
    MOVE_KNIGHT_PROMO_CAP,
    MOVE_BISHOP_PROMO_CAP,
    MOVE_ROOK_PROMO_CAP,
    MOVE_QUEEN_PROMO_CAP,
};

/* Castling bitfield */
enum {
    CASTLE_K = 1,
    CASTLE_Q = 2,
    CASTLE_k = 4,
    CASTLE_q = 8,
};

/* Board state */
typedef struct {
    Bitboard pieces[6][2];
    int side;
    Square ep;
    uint8_t castling;
    uint8_t halfmove;
    uint16_t fullmove;
} Board;

/* Move encoded as uint16: [from:6][to:6][flag:4] */
typedef uint16_t Move;

/* ========================================================================== */
/* MOVE ENCODING MACROS                                                       */
/* ========================================================================== */

#define MOVE_GET_FROM(m)  ((Square)((m) & 0x3F))
#define MOVE_GET_TO(m)    ((Square)(((m) >> 6) & 0x3F))
#define MOVE_GET_FLAG(m)  ((int)(((m) >> 12) & 0xF))
#define MOVE_ENCODE(f,t,fl) ((Move)((f) | ((t) << 6) | ((fl) << 12)))

/* ========================================================================== */
/* CONSTANTS                                                                  */
/* ========================================================================== */

#define PAWN_WHITE_INITIAL 0x00FF000000000000ULL
#define PAWN_BLACK_INITIAL 0x000000000000FF00ULL
#define ROOK_WHITE_INITIAL 0x8100000000000000ULL
#define ROOK_BLACK_INITIAL 0x0000000000000081ULL
#define KNIGHT_WHITE_INITIAL 0x4200000000000000ULL
#define KNIGHT_BLACK_INITIAL 0x0000000000000042ULL
#define BISHOP_WHITE_INITIAL 0x2400000000000000ULL
#define BISHOP_BLACK_INITIAL 0x0000000000000024ULL
#define QUEEN_WHITE_INITIAL 0x1000000000000000ULL
#define QUEEN_BLACK_INITIAL 0x0000000000000010ULL
#define KING_WHITE_INITIAL 0x0800000000000000ULL
#define KING_BLACK_INITIAL 0x0000000000000008ULL

#define FILE_A 0x8080808080808080ULL
#define FILE_B 0x4040404040404040ULL
#define FILE_C 0x2020202020202020ULL
#define FILE_D 0x1010101010101010ULL
#define FILE_E 0x0808080808080808ULL
#define FILE_F 0x0404040404040404ULL
#define FILE_G 0x0202020202020202ULL
#define FILE_H 0x0101010101010101ULL

#define RANK_1 0xFF00000000000000ULL
#define RANK_2 0x00FF000000000000ULL
#define RANK_3 0x0000FF0000000000ULL
#define RANK_4 0x000000FF00000000ULL
#define RANK_5 0x00000000FF000000ULL
#define RANK_6 0x0000000000FF0000ULL
#define RANK_7 0x000000000000FF00ULL
#define RANK_8 0x00000000000000FFULL

#define NOT_A_FILE 0x7F7F7F7F7F7F7F7FULL
#define NOT_H_FILE 0xFEFEFEFEFEFEFEFEULL
#define NOT_AB_FILE 0x3F3F3F3F3F3F3F3FULL
#define NOT_GH_FILE 0xFCFCFCFCFCFCFCFCULL

/* ========================================================================== */
/* BITBOARD UTILITY INLINES                                                   */
/* ========================================================================== */

static inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
static inline int lsb_idx(Bitboard b)  { return __builtin_ctzll(b); }
static inline int msb_idx(Bitboard b)  { return 63 - __builtin_clzll(b); }
static inline Bitboard lsb(Bitboard b) { return b & -b; }
static inline void clear_bit(Bitboard *b, Square sq) { *b &= ~(1ULL << sq); }
static inline void set_bit(Bitboard *b, Square sq)   { *b |= (1ULL << sq); }
static inline bool get_bit(Bitboard b, Square sq)    { return (b >> sq) & 1; }
static inline Bitboard sq_bb(Square sq)              { return 1ULL << sq; }
static inline int file_of(Square sq)                 { return 7 - (sq & 7); }
static inline int rank_of(Square sq)                 { return sq >> 3; }
static inline Square make_sq(int file, int rank) {
    return (Square)((rank << 3) | (7 - file));
}

/* ========================================================================== */
/* BOARD QUERIES                                                              */
/* ========================================================================== */

static inline Bitboard board_occ_color(const Board *b, int color) {
    Bitboard occ = 0;
    for (int pt = 0; pt < 6; pt++) occ |= b->pieces[pt][color];
    return occ;
}
static inline Bitboard board_occ(const Board *b) {
    return board_occ_color(b, WHITE) | board_occ_color(b, BLACK);
}
static inline int piece_on(const Board *b, Square sq) {
    Bitboard m = sq_bb(sq);
    for (int pt = 0; pt < 6; pt++)
        if (b->pieces[pt][WHITE] & m) return pt;
    for (int pt = 0; pt < 6; pt++)
        if (b->pieces[pt][BLACK] & m) return pt;
    return PIECE_NONE;
}
static inline Square king_sq(const Board *b, int color) {
    return lsb_idx(b->pieces[PIECE_KING][color]);
}

/* ========================================================================== */
/* FUNCTION DECLARATIONS                                                      */
/* ========================================================================== */

void init_tables(void);
Board board_default(void);
void board_from_fen(Board *b, const char *fen);
void board_to_fen(const Board *b, char *fen);

void print_bitboard(Bitboard bb);
void print_board(const Board *b);

Bitboard pawn_attacks(Square sq, int color);
Bitboard knight_attacks(Square sq);
Bitboard king_attacks(Square sq);
Bitboard bishop_attacks(Square sq, Bitboard occ);
Bitboard rook_attacks(Square sq, Bitboard occ);
Bitboard queen_attacks(Square sq, Bitboard occ);
Bitboard piece_attacks(int piece, Square sq, Bitboard occ, int color);

bool is_sq_attacked(const Board *b, Square sq, int by_color);
bool is_check(const Board *b);
bool is_checkmate(const Board *b);
bool is_stalemate(const Board *b);

int gen_pseudo_moves(const Board *b, Move *moves);
int gen_legal_moves(const Board *b, Move *moves);

void apply_move(Board *b, Move move);

void board_to_planes(const Board *b, uint8_t planes[14][8][8]);

int move_to_uci(Move m, char *s);
int uci_parse_move(const Board *b, const char *s, Move *m);

#endif
