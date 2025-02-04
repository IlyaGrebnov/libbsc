SHELL = /bin/sh
OS := $(shell uname -s)
ARCH := $(shell uname -m)
MACHINE_ARCH := $(shell uname -p)

ifeq ($(OS),Darwin)
    LIBEXT = dylib
else
    LIBEXT = so
endif

CC ?= clang
CXX ?= clang++
AR ?= ar
RANLIB ?= ranlib

CFLAGS ?= -g -Wall

# Comment out CFLAGS line below for compatability mode for 32bit file sizes
# (less than 2GB) and systems that have compilers that treat int as 64bit
# natively (ie: modern AIX)
CFLAGS += -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64

# Comment out CFLAGS line below to disable optimizations
CFLAGS += -O3 -fomit-frame-pointer -fstrict-aliasing -ffast-math

# Comment out CFLAGS line below to disable AVX2 instruction set (performance will suffer)
ifneq (,$(filter x86_64 i686 i386,$(ARCH)))
    CFLAGS += -mavx2
endif

# Comment out CFLAGS line below to disable OpenMP optimizations
OPENMP ?= 0
ifneq (,$(filter 1 true yes,$(OPENMP)))
    CFLAGS += -fopenmp -DLIBBSC_OPENMP_SUPPORT
endif

# Comment out CFLAGS line below to enable debug output
DEBUG ?= 0
ifneq (,$(filter 0 false no,$(DEBUG)))
    CFLAGS += -DNDEBUG
endif

# Comment out CFLAGS line below to disable Sort Transform
CFLAGS += -DLIBBSC_SORT_TRANSFORM_SUPPORT

# Comment out CFLAGS line below to disable unaligned memory access
CFLAGS += -DLIBBSC_ALLOW_UNALIGNED_ACCESS

# Where you want bsc installed when you do 'make install'
PREFIX ?= /usr

NATIVE ?= 0
ifneq (,$(filter 1 true yes,$(NATIVE)))
    ifneq (,$(filter arm% aarch64,$(ARCH)))
        CFLAGS += -mcpu=native
    else ifneq (,$(filter powerpc,$(MACHINE_ARCH)))
        CFLAGS += -mtune=native
    else
        CFLAGS += -march=native
    endif
endif

OBJS = \
       adler32.o       \
       libsais.o       \
       coder.o         \
       qlfc.o          \
       qlfc_model.o    \
       detectors.o     \
       preprocessing.o \
       libbsc.o        \
       lzp.o           \
       platform.o      \
       st.o            \
       bwt.o           \

all: libbsc.a bsc libbsc.$(LIBEXT)

bsc: libbsc.a bsc.cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) bsc.cpp -o $@ $<

libbsc.a: $(OBJS)
	rm -f $@
	$(AR) cq $@ $(OBJS)
	@if ( test -f $(RANLIB) -o -f /usr/bin/ranlib -o \
		-f /bin/ranlib -o -f /usr/ccs/bin/ranlib ) ; then \
		echo $(RANLIB) $@ ; \
		$(RANLIB) $@ ; \
	fi

libbsc.$(LIBEXT): $(OBJS)
	rm -f $@
	$(CC) $(CFLAGS) -fPIC -shared $(LDFLAGS) -o $@ $(OBJS)

install: libbsc.a bsc libbsc.$(LIBEXT)
	if ( test ! -d $(DESTDIR)$(PREFIX)/bin ) ; then mkdir -p $(DESTDIR)$(PREFIX)/bin ; fi
	if ( test ! -d $(DESTDIR)$(PREFIX)/lib ) ; then mkdir -p $(DESTDIR)$(PREFIX)/lib ; fi
	if ( test ! -d $(DESTDIR)$(PREFIX)/include ) ; then mkdir -p $(DESTDIR)$(PREFIX)/include ; fi
	cp -f bsc $(DESTDIR)$(PREFIX)/bin/bsc
	chmod a+x $(DESTDIR)$(PREFIX)/bin/bsc
	cp -f libbsc/libbsc.h $(DESTDIR)$(PREFIX)/include
	chmod a+r $(DESTDIR)$(PREFIX)/include/libbsc.h
	cp -f libbsc.a $(DESTDIR)$(PREFIX)/lib
	chmod a+r $(DESTDIR)$(PREFIX)/lib/libbsc.a
	cp -f libbsc.$(LIBEXT) $(DESTDIR)$(PREFIX)/lib
	chmod a+r $(DESTDIR)$(PREFIX)/lib/libbsc.$(LIBEXT)

clean:
	rm -rf *.o libbsc.a libbsc.$(LIBEXT) bsc bsc.dSYM

adler32.o: libbsc/adler32/adler32.cpp
	$(CXX) $(CFLAGS) -c libbsc/adler32/adler32.cpp

libsais.o: libbsc/bwt/libsais/libsais.c
	$(CC) $(CFLAGS) -c libbsc/bwt/libsais/libsais.c

coder.o: libbsc/coder/coder.cpp
	$(CXX) $(CFLAGS) -c libbsc/coder/coder.cpp

qlfc.o: libbsc/coder/qlfc/qlfc.cpp
	$(CXX) $(CFLAGS) -c libbsc/coder/qlfc/qlfc.cpp

qlfc_model.o: libbsc/coder/qlfc/qlfc_model.cpp
	$(CXX) $(CFLAGS) -c libbsc/coder/qlfc/qlfc_model.cpp

detectors.o: libbsc/filters/detectors.cpp
	$(CXX) $(CFLAGS) -c libbsc/filters/detectors.cpp

preprocessing.o: libbsc/filters/preprocessing.cpp
	$(CXX) $(CFLAGS) -c libbsc/filters/preprocessing.cpp

libbsc.o: libbsc/libbsc/libbsc.cpp
	$(CXX) $(CFLAGS) -c libbsc/libbsc/libbsc.cpp

lzp.o: libbsc/lzp/lzp.cpp
	$(CXX) $(CFLAGS) -c libbsc/lzp/lzp.cpp

platform.o: libbsc/platform/platform.cpp
	$(CXX) $(CFLAGS) -c libbsc/platform/platform.cpp

st.o: libbsc/st/st.cpp
	$(CXX) $(CFLAGS) -c libbsc/st/st.cpp

bwt.o: libbsc/bwt/bwt.cpp
	$(CXX) $(CFLAGS) -c libbsc/bwt/bwt.cpp
