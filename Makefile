CC = g++
LD = g++

CFLAGS =   $(SFLAGS) -O2 -g -fomit-frame-pointer -ffast-math -Wall 
LDFLAGS =  $(SFLAGS) -O2 -g -Wall

all: llsvm-predict llsvm-train

llsvm-predict: llsvm-predict.c svm.o llsvm.o
	$(LD) $(LDFLAGS) -o llsvm-predict llsvm-predict.c svm.o llsvm.o

llsvm-train: llsvm-train.c svm.o llsvm.o
	$(LD) $(LDFLAGS) -o llsvm-train llsvm-train.c svm.o llsvm.o
	
llsvm.o: llsvm.cpp llsvm.h
	$(CXX) $(CFLAGS) -c llsvm.cpp
	
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp

clean:
	rm llsvm-predict llsvm-train *.o
	
	