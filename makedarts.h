#!/bin/bash

for i in {0..11}
do
	./face.a training/dart$i.jpg
	cp output.jpg darts_o/output_$i.jpg
done
