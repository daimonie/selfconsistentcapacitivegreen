#!/bin/bash
echo "Performing Coulomb Island calculations. See coulombMetaData.md. ";

rm *.txt;

python coulombIslandOccupation.py >> "data.txt";
python coulombIslandTransport.py -f "data.txt" >> "transport.txt";
python coulombIslandConductance.py -f "data.txt" >> "conductance.txt";
python coulombIslandConductanceMax.py -f "data.txt" >> "max.txt";
python coulombIslandConductanceRound.py -f "data.txt" >> "round.txt";
python coulombIslandConductanceFollow.py -f "data.txt" -n 0 >> "follow0.txt";
python coulombIslandConductanceFollow.py -f "data.txt" -n 1 >> "follow1.txt";
python coulombIslandConductanceFollow.py -f "data.txt" -n 2 >> "follow2.txt";
python coulombIslandConductanceFollow.py -f "data.txt" -n 3 >> "follow3.txt";

echo "Completed Coulomb Island calculations.";
