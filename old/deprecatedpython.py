
	numericalTolerance = 1e-3;
	numericalMethod = 'SLSQP'; #Sequential Least Squares Programming
	numericalConstraints = [];

	if len(nullSpace) == 2:
		#We have vectors that SPAN the nullspace, but we still need to find it.
		firstNullVector = matrix2numpy(nullSpace[0]);
		secondNullVector = matrix2numpy(nullSpace[1]);

		vector = lambda p: p[0] * firstNullVector + p[1] * secondNullVector

		numericalConstraints.append({
			'type': 'ineq', # fun needs to equal zero; This normalises the minimised vector
			'fun': lambda p: np.sum([ lambda q: -(q<0)*1.0    for q in vector(p)])
		}); 

		result = minimize( lambda p: np.sum(vector(p))-1, np.zeros((2)), method=numericalMethod, constraints=numericalConstraints, tol=numericalTolerance);
		selfConsistentProbabilityVector = vector(result.x); 
		print selfConsistentProbabilityVector;

	elif len(nullSpace) == 1:
		selfConsistentProbabilityVector = matrix2numpy(nullSpace);
	else:
		useMinimisation = 1;


	if useMinimisation==1:
		#
		numericalConstraints.append({
		'type': 'eq', # fun needs to equal zero; This normalises the minimised vector
		'fun': lambda p: np.sum(p)-1 
		}); 
		#Vector needs to be positive
		numericalConstraints.append({
			'type': 'ineq', # fun needs to be non negative
			'fun': lambda p: p 
		}); 
		print >> sys.stderr, "Initial guess: %.5f %.5f %.5f %.5f" % (initialGuess[0],initialGuess[1],initialGuess[2],initialGuess[3]);
		print >> sys.stderr, "Initial Error: %.3f" % numberError(initialGuess);
	 
		result = minimize( numberError, initialGuess, method=numericalMethod, constraints=numericalConstraints, tol=numericalTolerance);
		selfConsistentProbabilityVector = np.array(result.x); 

	raise Exception('abort');
	separationLength = 0;
	for i in range(4):
		separationLength += (initialGuess[i] - selfConsistentProbabilityVector[i])**2;

	print >> sys.stderr, "Final Error: %.3f" % numberError(selfConsistentProbabilityVector);
	print >> sys.stderr, "Self-consistent result: %.5f %.5f %.5f %.5f" % (selfConsistentProbabilityVector[0],selfConsistentProbabilityVector[1],selfConsistentProbabilityVector[2],selfConsistentProbabilityVector[3]);
	print >> sys.stderr, "Separation length: %.5f\n" % separationLength

	print "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f" % (betaIteration,beta,selfConsistentProbabilityVector[0],selfConsistentProbabilityVector[1],selfConsistentProbabilityVector[2],selfConsistentProbabilityVector[3], separationLength, bias, capacitive); 
