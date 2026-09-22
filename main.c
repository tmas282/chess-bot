#include "chess/chess.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "chess/minimax.h"

#define MAX_MOVES 256
#define LINE_BUF 4096
#define BOT_NAME "CassufasChess"
#define BOT_AUTHOR "Tomas-Moreira"

static Board board;

/* ========================================================================== */
/* Position command                                                            */
/* ========================================================================== */

static void parse_position(char *line) {
    /* Skip "position " */
    line += 9;
    while (*line == ' ') line++;

    if (strncmp(line, "startpos", 8) == 0) {
        board = board_default();
        line += 8;
    } else if (strncmp(line, "fen", 3) == 0) {
        line += 3;
        while (*line == ' ') line++;
        board_from_fen(&board, line);
        /* Advance past the FEN string (6 space-delimited fields) */
        int fields = 0;
        while (*line && fields < 6) {
            if (*line == ' ') fields++;
            line++;
        }
    }

    /* Skip to "moves" */
    while (*line == ' ') line++;
    if (strncmp(line, "moves", 5) != 0) return;
    line += 5;
    while (*line == ' ') line++;

    /* Parse and apply each move */
    while (*line) {
        char token[8];
        int ti = 0;
        while (*line && *line != ' ' && ti < 6) token[ti++] = *line++;
        token[ti] = 0;
        if (!token[0]) break;
        while (*line == ' ') line++;

        Move m;
        if (uci_parse_move(&board, token, &m)) {
            apply_move(&board, m);
        }
    }
}

/* ========================================================================== */
/* Go command                                                                 */
/* ========================================================================== */

static void parse_go(char *line) {
    (void)line;
    Move m = minimax_init(&board);
    if (m == 0) {
        printf("bestmove 0000\n");
    } else {
        char best[4] = "0000";
        move_to_uci(m, best);
        printf("bestmove %s\n", best);
    }
    fflush(stdout);
}

/* ========================================================================== */
/* Main UCI loop                                                              */
/* ========================================================================== */

int main(void) {
    init_tables();
    setbuf(stdin, NULL);
    setbuf(stdout, NULL);

    char line[LINE_BUF];
    fflush(stdin);
    while (fgets(line, sizeof(line), stdin)) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;

        if (strcmp(line, "uci") == 0) {
            printf("id name %s\n", BOT_NAME);
            printf("id author %s\n", BOT_AUTHOR);
            printf("uciok\n");
            fflush(stdout);
        } else if (strcmp(line, "isready") == 0) {
            printf("readyok\n");
            fflush(stdout);
        } else if (strcmp(line, "ucinewgame") == 0) {
            board = board_default();
        } else if (strncmp(line, "position", 8) == 0) {
            parse_position(line);
        } else if (strncmp(line, "go", 2) == 0) {
            parse_go(line);
        } else if (strcmp(line, "stop") == 0) {
            /* No-op since we don't search asynchronously */
        } else if (strcmp(line, "quit") == 0) {
            break;
        }
    }
    return 0;
}
