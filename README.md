# Projectively equivariant neural networks

This repository contains code used for the experiments in the paper 
```
@article{bokman2023projEqNetwork,
  title={In Search of Projectively Equivariant Networks},
  author={Bökman, Georg and Flinth, Axel and Kahl, Fredrik},
  journal={arXiv:2209.14719v2},
  year={2023}
```
It is mainly released for transparency and reproducibility purposes. Should you however find this code useful for your research, please cite us!

### Spinor Field Networks
Simply run the provided jupyter notebook.

### Vierernet
To run an experiment, run viererconv.py with arguments number of epochs per experiment, number of experiment, dataset name ('mnist', 'cifar_single' or 'cifar_double') and directory to save in (optional). For example, to run 
the mnist experiment as done in the paper, run

```
    python viererconv.py 100 30 mnist 
```
 
The results will be saved in the directory 'directory name'. Since it was not specified here, it will default to the 'res_'+ dataset, i.e 'res_mnist'. The data will be automatically downloaded. If a gpu is available, it will be used.

To evaluate the experiments, run evalscript.py with parameters number of epochs, number of runs, name of directory of result and dataset name ('mnist, 'cifar_single' or 'cifar_double')
(dataset name is optional, just for plotting purposes). For example, to evaluate the mnist experiment as ran above, write

```
   python evalscript.py 100 30 res_mnist mnist
```
The graphics will be saved in the repository 'plots' with an obvious name. The p-value that is displayed is a result of a simple non-parametric test. It will not be very meaningful if the number of 
experiments is very low. Also, the p-value is for the vierernet outperforming the baseline. A p-value close to 1 means that the baseline is outperforming the vierernet.

## Copyright

The code is released under a Creative Commons Attribution-ShareAlike (CC BY-SA 2.0) licence. You are hence free to copy, re-distribute and modify the code as you please, as long as you credit us and use the same licence for your derived work.
