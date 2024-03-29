# Author: Decebal Constantin Mocanu et al.;
# Plot performance of all three models on CIFAR10

# This is a pre-alpha free software and was tested with Python 3.5.2, Keras 2.1.3, Keras_Contrib 0.0.2, Tensorflow 1.5.0, Numpy 1.14;
# The code is distributed in the hope that it may be useful, but WITHOUT ANY WARRANTIES; The use of this software is entirely at the user's own risk;
# For an easy understanding of the code functionality please read the following articles.

# If you use parts of this code please cite the following articles:
#@article{Mocanu2018SET,
#  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#  journal =       {Nature Communications},
#  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#  year =          {2018},
#  doi =           {10.1038/s41467-018-04316-3}
#}

#@Article{Mocanu2016XBM,
#author="Mocanu, Decebal Constantin and Mocanu, Elena and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio",
#title="A topological insight into restricted Boltzmann machines",
#journal="Machine Learning",
#year="2016",
#volume="104",
#number="2",
#pages="243--270",
#doi="10.1007/s10994-016-5570-z",
#url="https://doi.org/10.1007/s10994-016-5570-z"
#}

#@phdthesis{Mocanu2017PhDthesis,
#title = "Network computations in artificial intelligence",
#author = "D.C. Mocanu",
#year = "2017",
#isbn = "978-90-386-4305-2",
#publisher = "Eindhoven University of Technology",
#}

import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(6,5))
plt.xlabel("Epochs[#]")
plt.ylabel("CIFAR10\nAccuracy [%]")
plt.ylim(0, 100)

e1=np.loadtxt("results_basebig/dense_mlp_srelu_sgd_cifar10_acc.txt")
e2=np.loadtxt("results_basebig/fixprob_mlp_srelu_sgd_cifar10_acc.txt")
e3=np.loadtxt("results_basebig/set_mlp_srelu_sgd_cifar10_acc.txt")
e4=np.loadtxt("results_base/dense_mlp_srelu_sgd_cifar10_acc.txt")
e5=np.loadtxt("results_base/fixprob_mlp_srelu_sgd_cifar10_acc.txt")
e6=np.loadtxt("results_base/set_mlp_srelu_sgd_cifar10_acc.txt")
e7=np.loadtxt("results_new50_2/set2_mlp_srelu_sgd_cifar10_acc.txt")
e8=np.loadtxt("results_new10_2/set2_mlp_srelu_sgd_cifar10_acc.txt")

plt.plot(e1*100,label="Dense (all)",  linewidth=0.7)
plt.plot(e2*100,label="Fix (all)",  linewidth=0.7)
plt.plot(e3*100,label="SET (all)",  linewidth=0.7)
plt.plot(e4*100,label="Dense (sample)",  linewidth=0.7)
plt.plot(e5*100,label="Fix (sample)",  linewidth=0.7)
plt.plot(e6*100,label="SET (sample)",  linewidth=0.7)
plt.plot(e7*100,label="SPET (sample) 20 gens" , linewidth=0.7)
plt.plot(e8*100,label="SPET (sample) 100 gens",  linewidth=0.7)
plt.legend(bbox_to_anchor=(1, 1), fontsize = 'x-small')
plt.grid(True)
plt.tight_layout()
plt.savefig("cifar10_models_performance_bb.pdf")
plt.close()



plt.figure(figsize=(4.5,5))
plt.ylim(0, 100)
plt.xlabel("Epochs[#]")
plt.ylabel("CIFAR10\nAccuracy [%]")

e1=np.loadtxt("results_final/dense_mlp_srelu_sgd_cifar10_acc.txt")
e2=np.loadtxt("results_final/fixprob_mlp_srelu_sgd_cifar10_acc.txt")
e3=np.loadtxt("results_final/set_mlp_srelu_sgd_cifar10_acc.txt")
e4=np.loadtxt("results_final/set2_mlp_srelu_sgd_cifar10_acc.txt")

plt.plot(e1*100,label="Dense (sample)",  linewidth=0.7)
plt.plot(e2*100,label="Fix (sample)",  linewidth=0.7)
plt.plot(e3*100,label="SET (sample)",  linewidth=0.7)
plt.plot(e4*100,label="SPET (sample) 100 gens",  linewidth=0.7)
plt.legend(bbox_to_anchor=(1, 1), fontsize = 'x-small')
plt.grid(True)
plt.tight_layout()
plt.savefig("cifar10_models_performance_sb.pdf")
plt.close()



plt.figure(figsize=(6,5))
plt.ylim(0, 100)
plt.xlabel("Epochs[#]")
plt.ylabel("CIFAR10\nAccuracy [%]")

e1=np.loadtxt("results_final/dense_mlp_srelu_sgd_cifar10_acc_tr.txt")
e2=np.loadtxt("results_final/fixprob_mlp_srelu_sgd_cifar10_acc_tr.txt")
e3=np.loadtxt("results_final/set_mlp_srelu_sgd_cifar10_acc_tr.txt")
e4=np.loadtxt("results_final/set2_mlp_srelu_sgd_cifar10_acc_tr.txt")

plt.plot(e1*100,label="Dense (sample)",  linewidth=0.7)
plt.plot(e2*100,label="Fix (sample)",  linewidth=0.7)
plt.plot(e3*100,label="SET (sample)",  linewidth=0.7)
plt.plot(e4*100,label="SPET (sample) 100 gens",  linewidth=0.7)
plt.legend(bbox_to_anchor=(1, 1), fontsize = 'x-small')
plt.grid(True)
plt.tight_layout()
plt.savefig("cifar10_models_performance_sb_tr.pdf")
plt.close()
