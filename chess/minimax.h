#ifndef MINIMAX_H
#define MINIMAX_H

#define DEPTH 4
#include "chess.h"

float minimax(Board* board_p, unsigned char depth, bool maximizer);
Move minimax_init(Board* board_p);
Board clone_board_and_move(Board* board_p, Move* move);

#endif