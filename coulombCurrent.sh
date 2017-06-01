#!/bin/bash
python coulombIslandCurrent.py -b .001 >> current.txt
python coulombIslandCurrent.py -b 0.1 >> current2.txt
python coulombIslandCurrent.py -b 1.0 >> current3.txt
python coulombIslandCurrent.py -b 10.0 >> current4.txt
