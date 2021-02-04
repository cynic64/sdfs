CC=cc
EXTRA=
CFLAGS=-O2 -Wall -Wextra -pedantic -Wno-sign-compare $(EXTRA)
LIBS=-lvulkan -lglfw -lm

main: main.c shaders external/
	$(CC) $(CFLAGS) $(LIBS) main.c -o main

shaders: shaders/*
	./compile_shaders.pl

.PHONY: clean

clean:
	rm main
