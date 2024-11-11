CC=clang
CFLAGS=-Wall -Wextra -c

EXES = opssfull opssaic opssmov opsswhite

.PHONY: all clean

all: $(EXES)

opssfull : opssfull.o MARgetaic.o MARfit.o EEGrealA.o MARgetpfpb.o matmisc.o matmisc2.o matmisc5.o
	cc -lm -o opssfull opssfull.o MARgetaic.o MARfit.o EEGrealA.o MARgetpfpb.o matmisc.o matmisc2.o matmisc5.o
opssfull.o : opssfull.c EEGdef.h EEGmat.h
	cc -c opssfull.c

opsswhite : opsswhite.o MAR_residual.o MARfit.o EEGrealA.o MARgetpfpb.o matmisc.o matmisc2.o matmisc5.o
	cc -lm -o opsswhite opsswhite.o MAR_residual.o MARfit.o EEGrealA.o MARgetpfpb.o matmisc.o matmisc2.o matmisc5.o

opssmov : opssmov.o MARfit.o EEGrealA.o MARgetpfpb.o matmisc.o matmisc2.o matmisc5.o EEG_tpool.o
	cc -lm -o opssmov opssmov.o MARfit.o EEGrealA.o MARgetpfpb.o matmisc.o matmisc2.o matmisc5.o EEG_tpool.o
opssmov.o : opssmov.c EEGdef.h EEGmat.h
	cc -c opssmov.c

opssaic : opssaic.o MARgetaic.o MARfit.o EEGrealA.o MARgetpfpb.o matmisc.o matmisc2.o matmisc5.o
	cc -lm -o opssaic opssaic.o MARgetaic.o MARfit.o EEGrealA.o MARgetpfpb.o matmisc.o matmisc2.o matmisc5.o
opssaic.o : opssaic.c EEGdef.h EEGmat.h
	cc -c opssaic.c

MARgetaic.o : MARgetaic.c
	cc -c MARgetaic.c
MARfit.o : MARfit.c EEGdef.h EEGmat.h
	cc -c MARfit.c
EEGrealA.o : EEGrealA.c EEGdef.h EEGmat.h
	cc -c EEGrealA.c
MARgetpfpb.o : MARgetpfpb.c EEGdef.h EEGmat.h
	cc -c MARgetpfpb.c
MAR_residual.o : MAR_residual.c EEGmat.h
	cc -c MAR_residual.c
matmisc.o : matmisc.c
	cc -c matmisc.c
matmisc2.o : matmisc2.c
	cc -c matmisc2.c
matmisc5.o : matmisc5.c
	cc -c matmisc5.c

clean:
	rm -f $(EXES) *.o
