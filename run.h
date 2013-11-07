#!/bin/bash
# $1- input | $2 - output | `pkg-config --cflags --libs opencv`
# for zshell $=O_LIBS
#g++ $1 -o $2 $O_LIBS  
#./$2 $3 $4 $5 $6
make
make run
