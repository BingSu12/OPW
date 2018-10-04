#############################################################################
# # # Order-preserving Optimal Transport for Distances between Sequences# # #
#############################################################################

1. Introduction.

This package includes the propotype MATLAB codes for computing the Order-Preserving Wasserstein Distance (OPW) and the Temporally Coupled Optimal Transport (TCOT) distance, described in

	"Order-preserving Optimal Transport for Distances between Sequences"
	Bing Su and Gang Hua. TPAMI, 2018, DOI: 10.1109/TPAMI.2018.2870154.

	"Order-preserving Wasserstein Distance for Sequence Matching"
	Bing Su and Gang Hua. CVPR, 2017, pp. 2906-2914.

----------------------------------------------------------------------

Tested under windows 7 x64, matlab R2015b.

############################################################################

2. License & disclaimer.

    The codes can be used for research purposes only. This package is strictly for non-commercial academic use only.

############################################################################

3.  Usage & Dependency.

This package contains prototype versions for computing the OPW distance and the TCOT distance in Matlab R2015b.
  
- OPW.m --- computing the OPW distance between two sequences X and Y

- OPW_w.m --- computing the OPW distance between two sequences X and Y, where the weights of vectors can be specified.

- TCOT.m --- computing the TCOT distance between two sequences X and Y


Dependency:

"TCOT.m" depends on the following code to perform the Sinkhorn's matrix scaling

- sinkhornTransport.m

by Marco Cuturi; this code can be downloaded from the website: http://marcocuturi.net/SI.html

**For convenience, we also include this code in this package, but please check the licence in http://marcocuturi.net/SI.html if you want to make use of this code.

############################################################################

4. Notice

1) We use these distances as foundations of distance-based classifiers such as the k-NN classifier to perform sequence classification in the paper.

2) The default parameters in this package are adjusted on the datasets used in the paper. You may need to adjust the parameters when applying it on a new dataset. Note that With some parameters, some entries of K may exceed the maching-precision limit; in such cases, you may need to adjust the parameters, and/or normalize the input features in sequences or the matrix D; Please see the paper for details.

3) We utilized the code "sinkhornTransport.m" provided by Marco Cuturi which is publicly available. Please check the licence of it if you want to make use of this code.

############################################################################

5. Citations

Please cite the following papers if you use the codes:

1) Bing Su and Gang Hua, "Order-preserving Optimal Transport for Distances between Sequences," IEEE Trans. Pattern Anal.Mach. Intell., 2018, DOI: 10.1109/TPAMI.2018.2870154.

2) Bing Su and Gang Hua, "Order-preserving Wasserstein Distance for Sequence Matching," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2017, pp. 2906¨C2914.


///////////////////////////////////////////////////
