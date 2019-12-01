convert: run
	ffmpeg -y -loglevel 0 -i test.ppm test.png

run: compile
	./smoke.out

compile: main.cpp
	g++ main.cpp -o smoke.out