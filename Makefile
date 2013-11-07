all:
	g++ face.cpp -o face.a $(O_LIBS)

run:
	./face.a
