SDL2_CFLAGS = `pkgconf --cflags sdl2`
SDL2_LIBS = `pkgconf --libs sdl2`
gui:
	cc -Wall -Wextra -Ofast $(SDL2_CFLAGS) $(SDL2_LIBS) -o nanopond nanopond.c -lpthread

clean:
	rm -f *.o nanopond *.dSYM
