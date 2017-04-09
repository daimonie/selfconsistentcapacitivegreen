It strikes me as rather silly that I did not write down how to use these files. Let's go over them.

In principle, the full calculations are performed by 'coulombIslandCalculations.sh'. However, more instructions might be needed. Step by step:

1. python coulombIslandOccupation.py >> data.txt
	
	This calculates the occupational probabilities for the coulomb Isalnd at different temperatures.

	python occupationPlot.py -f "data.txt"  allows for plotting the figure.

2. python coulombIslandTransport.py -f "data.txt" >> "transport.txt"
	
	This calculates T(E) at the different temperatures.

	python contourPlot.py -f "transport.txt" allows for plotting the (beta, epsilon, T(e,beta)) 
	contour map.
	python transportPlot.py -f "transport.txt" allows for plotting a few of the lines.

3. python coulombIslandConductance.py -f "data.txt" >> "conductance.txt"

	Calculates the average conductance in -4 U < eps < 4 U. This averaging effect gives a very smooth graph. Might be reproducible by using a smaller bound.

	python conductancePlot.py -f "conductance.txt" allows for inspecting the average conductance versus temperature, showing the Kondo effect.

4. python coulombIslandConductanceMax.py -f "data.txt" >> "max.txt"
	
	Calculates the maximum conductance in -4 U < eps < 4 U. This shows a rather sharp Kondo valley.

	python conductanceMaxPlot.py -f "max.txt" allows for inspecting the maximum conductance versus temperature.
5. python coulombIslandConductanceRound.py -f "data.txt" >> "round.txt"
	
	Calculates the average conductance in -1e-2 U < eps < 1e-2 U.  

	python conductanceRoundPlot.py -f "round.txt" allows for inspecting the maximum conductance versus temperature.
5. python coulombIslandConductanceFollow.py -f "data.txt" -n 0>> "follow0.txt"
	
	Calculates the conductance around one of the eigen values.

	python conductanceFollowPlot.py -f "follow0.txt" allows for inspecting the maximum conductance around an eigenvalue versus temperature.