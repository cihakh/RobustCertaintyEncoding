# RobustCertaintyEncoding

## This repository contains sample code used to generate some figures and results in the paper entitled ``Robustly encoding certainty in a neural circuit model'' by Heather L Cihak and Zachary P Kilpatrick (citation/link to be incorporated) with the abstract as follows below.


### ABSTRACT:
Localized persistent neural activity can encode delayed estimates of continuous variables. Common experiments require that subjects store and report the feature value (e.g., orientation) of a particular cue (e.g., oriented bar on a screen) after a delay. Visualizing recorded activity of neurons along their feature tuning reveals activity bumps whose centers wander stochastically, degrading the estimate over time. 
    Bump position therefore represents the remembered estimate. Recent work suggests bump amplitude may represent estimate certainty reflecting a probabilistic population code for a Bayesian posterior. Idealized models of this type are fragile due to the fine tuning common to constructed continuum attractors in dynamical systems. Here we propose an alternative metastable model for robustly supporting multiple bump amplitudes by extending neural circuit models to include quantized nonlinearities.
    Asymptotic projections of circuit activity produce low-dimensional evolution equations for the amplitude and position of bump solutions in response to external stimuli and noise perturbations. Analysis of reduced equations accurately characterizes phase variance and the dynamics of amplitude transitions between stable discrete values. More salient cues generate bumps of higher amplitude which wander less, consistent with the experimental finding that greater certainty correlates with more accurate memories.

## Summary of files
First drafts of the sample code used to generate figures is now posted along with txt files containing sample Monte Carlo simulation data.
### The main functions utilized can be found in the file 'Encoding_Certainty_Functions.'' Where possible variable names are chosen to reflect those within the paper. These function are written for the wizard-hat synaptic profiles chosen in the paper. If one wishes to change this or other conventions, then portions of these functions must be rewritten accordingly. 
### Examples using the functions and generating plots contained within the manuscript are presented within the notebooks ''SampleCode_Deterministic.ipynb'' and ''SampleCode_Stochastic.ipynb.'' The example notebook also contains several notes on the figures and particular sections of code. 
### txt files containing sample data for variance plots and a transition time plot. Note: ensure the path to these files is correct in the notebooks to generate the plot.

## Requirements
### All code was run using python 3.7. See the beginning of the Examples notebook for the specific packages utilized. Largely, NumPy and SciPy were utilized for calculations and matplotlib for plotting. tqdm was used to create progress bars and joblib for parallel computing. Note that tqdm and joblib are not necessary for the calculations themselves and can be removed from the code if one prefers. tqdm created useful progress bars and joblib aided in shortening the computation time for the 10,000 simulations needed for variance calculations. A user could use other methods if preferred.
### joblib requires the use of Python 3.7 or greater. More details on dependencies and installation can be found at https://joblib.readthedocs.io/en/stable/
### tqdm does not have dependencies as noted in its documnetation https://github.com/tqdm/tqdm

## Credits: 
### Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.
### Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.
### da Costa-Luis, (2019). tqdm: A Fast, Extensible Progress Meter for Python and CLI. Journal of Open Source Software, 4(37), 1277, https://doi.org/10.21105/joss.01277
### Varoquax G., (2011), joblib, GitHub repository, https://github.com/joblib/joblib

## Acknowledgements: Thank you so very much to all those above for developing and maintaining the open source packages/functions that I have utilized within my projects. Additionally, thank you to the platypus for being the most magnificent monotreme. 
