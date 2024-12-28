# bioinformatics-and-drug-discovery
In this repository we discuss and share all about the Bioinformatics and Drug Discovery field and here we share codding area as well.


Bioinformatics Project - Computational Drug Discovery Exploratory Data Analysis
------------------------------------------------------------------------------------------

In part we will be performing Descriptor Calculation and Exploratory Data Analysis.


Calculate Lipinski descriptors
--------------------------------------
	-> Christopher Lipinski, a scientist at Pfizer, came up with a set of rule-of-thumb for evaluating the druglikeness 
	   of compounds. Such druglikeness is based on the Absorption, Distribution, Metabolism and Excretion (ADME) that 
	   is also known as the pharmacokinetic profile.

	Lipinski's Rule.
	------------------
		-> Molecular weight < 500 Dalton
		-> Octanol-water partition coefficient (LogP) < 5
		-> Hydrogen bond donors < 5
		-> Hydrogen bond acceptors < 10


Bioinformatics Project - Computational Drug Discovery [Part 5] Comparing Regressors
--------------------------------------------------------------------------------------

Model Training and Evaluation with LazyRegressor
----------------------------------------------------
	clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
	-----------------------------------------------------------------------------
		-> LazyRegressor is a class from lazypredict library. It automatically trains and evaluates a variety of regression models without requiring you to explicitly define them.
	   	   models_train,predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)

		Parameters:
		-----------
			-> verbose=0: Disables verbose output.
			-> ignore_warnings=True: Suppresses warnings during model training.
			-> custom_metric=None: Uses default metrics for evaluation.

	Training and Predictions
	---------------------------
		models_train, predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
		models_test, predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)

			X_train: Features for training.
			X_test: Features for testing.
			Y_train: Target values for training.
			Y_test: Target values for testing.

	Purpose of the Workflow
	--------------------------
		Feature Selection:
		------------------
			-> Removes redundant or unimportant features to improve model performance and reduce overfitting.
	Data Splitting:
	---------------------
		-> Ensures the models are evaluated on unseen data, providing a realistic measure of performance.

	Model Comparison with LazyRegressor:
	-------------------------------------
		-> Quickly compares a variety of regression models to determine the best-performing one for the dataset.
		-> Evaluates both training performance and generalization to unseen data.
