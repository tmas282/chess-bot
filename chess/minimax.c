#include "chess.h"
#include "minimax.h"
#include <stdbool.h>

#define SCORE_EPS 0.0005f

float minimax(Board* board_p, unsigned char depth, bool side){
    if(is_checkmate(board_p)){
        if(side)
            return -1.0f - (float)depth * SCORE_EPS;
        else
            return 1.0f + (float)depth * SCORE_EPS;
    }
    else if(is_stalemate(board_p)){
        return 0;
    }
    else if(depth == 0){
        return 0; //TODO: implement heuristic call
    }
    else if(side == true){
        float r = -1.0f - (float)depth * SCORE_EPS;
        Move moves[256];
        short count = gen_legal_moves(board_p, moves);
        for (short i = 0; i < count; i++)
        {
            Board b = clone_board_and_move(board_p, &moves[i]);
            float possible = minimax(&b, depth-1, false);
            if(r < possible){
                r = possible;
            }
        }
        return r;
    }
    else if(side == false){
        float r = 1.0f + (float)depth * SCORE_EPS;
        Move moves[256];
        short count = gen_legal_moves(board_p, moves);
        for (short i = 0; i < count; i++)
        {
            Board b = clone_board_and_move(board_p, &moves[i]);
            float possible = minimax(&b, depth-1, true);
            if(r > possible){
                r = possible;
            }
        }
        return r;
    }
    return 0;
}

Move minimax_init(Board* board_p){
    if(is_checkmate(board_p) || is_stalemate(board_p)){
        return 0;
    }
    int side = board_p->side;
    float r;
    if(side == BLACK)
        r = 1.0f;
    if(side == WHITE)                       
        r = -1.0f;
    unsigned char indexBestMove = 0;
    Move moves[256];
    short count = gen_legal_moves(board_p, moves);
    for (short i = 0; i < count; i++)
    {
        Board b = clone_board_and_move(board_p, &moves[i]);
        float possible = minimax(&b, DEPTH, side == WHITE ? false : true);
        if(r > possible && side == BLACK){
            r = possible;
            indexBestMove = (unsigned char) i;
        }
        else if(r < possible && side == WHITE){
            r = possible;
            indexBestMove = (unsigned char) i;
        }
    }
    return moves[indexBestMove];
}

Board clone_board_and_move(Board* board_p, Move* move){
    char fen[92];
    board_to_fen(board_p, &fen[0]);
    Board new_board;
    board_from_fen(&new_board, &fen[0]);
    apply_move(&new_board, *move);
    return new_board;
}