SHELL = /bin/sh

CC = g++
AR = ar
RANLIB = ranlib

CFLAGS = -g -Wall -Ilibbsc

# Sort Transform is patented by Michael Schindler under US patent 6,199,064.
# However for research purposes this algorithm is included in this software.
# So if you are of the type who should worry about this (making money) worry away.
# The author shall have no liability with respect to the infringement of
# copyrights, trade secrets or any patents by this software. In no event will
# the author be liable for any lost revenue or profits or other special,
# indirect and consequential damages.

# Sort Transform is disabled by default and can be enabled by defining the
# preprocessor macro LIBBSC_SORT_TRANSFORM_SUPPORT at compile time.

#CFLAGS += -DLIBBSC_SORT_TRANSFORM_SUPPORT

# Comment out CFLAGS line below for compatability mode for 32bit file sizes
# (less than 2GB) and systems that have compilers that treat int as 64bit
# natively (ie: modern AIX)
CFLAGS += -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64

# Comment out CFLAGS line below to disable optimizations
CFLAGS += -O3 -fomit-frame-pointer -fstrict-aliasing -ffast-math

# Comment out CFLAGS line below to disable OpenMP optimizations
CFLAGS += -fopenmp

# Comment out CFLAGS line below to enable debug output
CFLAGS += -DNDEBUG

# Where you want bsc installed when you do 'make install'
PREFIX = /usr

OBJS = preprocessing.o \
       divsufsort.o    \
       detectors.o     \
       common.o        \
       libbsc.o        \
       crc32.o         \
       qlfc.o          \
       bwt.o           \
       lzp.o           \
       st.o

all: libbsc.a bsc

bsc: libbsc.a bsc.cpp
	$(CC) $(CFLAGS) bsc.cpp -o bsc -L. -lbsc

libbsc.a: $(OBJS)
	rm -f libbsc.a
	$(AR) cq libbsc.a $(OBJS)
	@if ( test -f $(RANLIB) -o -f /usr/bin/ranlib -o \
		-f /bin/ranlib -o -f /usr/ccs/bin/ranlib ) ; then \
		echo $(RANLIB) libbsc.a ; \
		$(RANLIB) libbsc.a ; \
	fi

install: libbsc.a bsc
	if ( test ! -d $(PREFIX)/bin ) ; then mkdir -p $(PREFIX)/bin ; fi
	if ( test ! -d $(PREFIX)/lib ) ; then mkdir -p $(PREFIX)/lib ; fi
	if ( test ! -d $(PREFIX)/include ) ; then mkdir -p $(PREFIX)/include ; fi
	cp -f bsc $(PREFIX)/bin/bsc
	chmod a+x $(PREFIX)/bin/bsc
	cp -f libbsc/libbsc.h $(PREFIX)/include
	chmod a+r $(PREFIX)/include/libbsc.h
	cp -f libbsc.a $(PREFIX)/lib
	chmod a+r $(PREFIX)/lib/libbsc.a

clean:
	rm -f *.o libbsc.a bsc

preprocessing.o: libbsc/filters/preprocessing.cpp
	$(CC) $(CFLAGS) -c libbsc/filters/preprocessing.cpp

divsufsort.o: libbsc/bwt/divsufsort/divsufsort.c
	$(CC) $(CFLAGS) -c libbsc/bwt/divsufsort/divsufsort.c

detectors.o: libbsc/filters/detectors.cpp
	$(CC) $(CFLAGS) -c libbsc/filters/detectors.cpp

common.o: libbsc/common/common.cpp
	$(CC) $(CFLAGS) -c libbsc/common/common.cpp

libbsc.o: libbsc/libbsc/libbsc.cpp
	$(CC) $(CFLAGS) -c libbsc/libbsc/libbsc.cpp

crc32.o: libbsc/crc32/crc32.cpp
	$(CC) $(CFLAGS) -c libbsc/crc32/crc32.cpp

qlfc.o: libbsc/qlfc/qlfc.cpp
	$(CC) $(CFLAGS) -c libbsc/qlfc/qlfc.cpp

bwt.o: libbsc/bwt/bwt.cpp
	$(CC) $(CFLAGS) -c libbsc/bwt/bwt.cpp

lzp.o: libbsc/lzp/lzp.cpp
	$(CC) $(CFLAGS) -c libbsc/lzp/lzp.cpp

st.o: libbsc/st/st.cpp
	$(CC) $(CFLAGS) -c libbsc/st/st.cpp
