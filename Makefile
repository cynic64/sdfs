CC=cc
EXTRA=
CFLAGS=-O2 -Wall -Wextra -pedantic -Wno-sign-compare -I. $(EXTRA)
LIBS=-lvulkan -lglfw -lm

main: shaders main.c external/ shaders/* external/render-c/* external/render-c/src/*
	$(CC) $(CFLAGS) $(LIBS) main.c -o main

shaders: shaders/*.glsl
	./compile_shaders.py

.PHONY: clean shaders

clean:
	rm main
