CC=g++
MODULES=face.o filters.o
CFLAGS=
all: faces

faces: face.o filters.o
	$(CC) $(CFLAGS) $(MODULES) -o face.a $(O_LIBS)

face.o: face.cpp
	$(CC) -c face.cpp

filters.o: filters.cpp
	$(CC) -c filters.cpp

run:
	./face.a
