all:
	clang++ -std=c++17 -Weverything -Wno-padded -Wno-c++98-compat -ferror-limit=1 -MP -MMD displacement.cpp -o displacement.o
