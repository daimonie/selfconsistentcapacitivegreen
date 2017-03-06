#!/bin/bash
echo "Performing Coulomb Island calculations.";

rm *.txt;

python coulombIslandOccupation.py >> "data.txt";
python coulombIslandTransport.py -f "data.txt" > "transport.txt";

echo "Completed Coulomb Island calculations.";
