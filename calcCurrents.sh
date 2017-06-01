#!/bin/bash
python coulombIslandCurrent.py -b .001 >> current.txt
python coulombIslandCurrent.py -b 0.1 >> current2.txt
python coulombIslandCurrent.py -b 1.0 >> current3.txt
python coulombIslandCurrent.py -b 10.0 >> current4.txt
python coulombIslandCurrent.py -b 100.0 >> current5.txt
python coulombIslandCurrent.py -b 1000.0 >> current6.txt

python perrinCurrent.py -b .00001 >> perrin.txt
python perrinCurrent.py -b .0001 >> perrin2.txt
python perrinCurrent.py -b .0001 >> perrin3.txt
python perrinCurrent.py -b .001 >> perrin4.txt
python perrinCurrent.py -b 0.1 >> perrin5.txt
python perrinCurrent.py -b 1.0 >> perrin6.txt
