install with following:
```
pip install git+https://github.com/Gilnore/causal-inference.git
```

This package uses a simple ball tree script for the knn methods of estimating mutual information and transfer entropy so the numba compiling time might be a bit long. 

references to papers:
1. Nagel, D., Diez, G. & Stock, G. Accurate estimation of the normalized mutual information of multidimensional data. The Journal of Chemical Physics 161, 054108 (2024).
2. Czyż, P., Grabowski, F., Vogt, J. E., Beerenwinkel, N. & Marx, A. Beyond Normal: On the Evaluation of Mutual Information Estimators. Preprint at https://doi.org/10.48550/arXiv.2306.11078 (2023).
3. Baboukani, P. S., Graversen, C., Alickovic, E. & Østergaard, J. Estimating Conditional Transfer Entropy in Time Series using Mutual Information and Non-linear Prediction. Entropy 22, 1124 (2020).
4. Kraskov, A., Stoegbauer, H. & Grassberger, P. Estimating Mutual Information. Phys. Rev. E 69, 066138 (2004).
5. Witter, J. & Houghton, C. Nearest-Neighbours Estimators for Conditional Mutual Information. Preprint at https://doi.org/10.48550/arXiv.2403.00556 (2024).
