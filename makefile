CC = gcc
CFLAGS = -Wall -Wextra -O2 -g
MATH_LIB = -lm

chess_bot: main.c chess/chess.c chess/chess.h chess/minimax.c chess/minimax.h
	$(CC) $(CFLAGS) main.c chess/chess.c chess/minimax.c -o chess_bot $(MATH_LIB)

test: chess/chess_test.c chess/chess.c chess/chess.h
	$(CC) $(CFLAGS) chess/chess_test.c chess/chess.c -o chess_test $(MATH_LIB)
	./chess_test
run:
	./chess_bot
debug: CFLAGS = -Wall -Wextra -O0 -g -DDEBUG
debug: chess_bot
	gdb ./chess_bot
	
.PHONY: clean
clean:
	rm -f chess_bot chess_test
