OPT ?= -O3

CFLAGS   = -std=gnu11 $(OPT) -Wall -Wextra -Wpedantic -Wstrict-aliasing -static

EXEC=lfsr merge mm sieve spmv


all: $(EXEC)

%: %.c *.h
	$(CC) -o $@ $< $(CFLAGS)

clean: 
	@rm -f $(EXEC)

gen_arr:
	python3 rand_c_arr.py --len=8192 --range=1000000
	python3 rand_spmv_arrs.py
