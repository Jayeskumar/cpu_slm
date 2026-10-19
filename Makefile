CC ?= cc
# PORTABLE=1 builds without CPU-specific instructions (for redistribution).
ifeq ($(PORTABLE),1)
ARCH =
else
ARCH = -march=native
endif
CFLAGS ?= -O3 $(ARCH) -std=c99 -Wall -Wextra -D_DEFAULT_SOURCE
LDLIBS = -lm

SRC = src/cortex.c src/lexicon.c
HDR = src/mnemo.h src/lexicon.h

all: build/mnemo build/libmnemo.so

build/mnemo: src/main.c $(SRC) $(HDR)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ src/main.c $(SRC) $(LDLIBS)

build/libmnemo.so: $(SRC) $(HDR)
	@mkdir -p build
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(SRC) $(LDLIBS)

clean:
	rm -rf build

.PHONY: all clean
