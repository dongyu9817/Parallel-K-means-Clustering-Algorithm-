## Parallel K-means Clustering Algorithm
### Author: Yu Dong
### SUMMARY
We are going to implement parallelized K-means clustering algorithm on both GPU and multi-core CPU platforms, and perform a detailed analysis of both performance characteristics.

### BACKGROUND
The K-Means clustering algorithm is an iterative algorithm that is widely used in the data mining area. It aims to partition dataset into K predefined distinct non-overlapping subgroups, called clusters. In there, each data point belongs to only one group. The algorithm assigns data points to a cluster to ensure that the sum of the squared distance between the data points and the cluster's centroid is at the minimum.

However, as the dataset becomes larger, the computation challenge that the K-Means algorithm facing becomes more obvious. For this reason, we decide to implement the K-Means algorithm in parallel using both the Cuda and OpenMp model to overcome problems of the simple K-Means algorithm model. We would also evaluate and compare the performance of the two models in terms of cluster quality, number of iteration, and elapsed time.
 
### CHALLENGES
The challenges of this project come from two parts. The first part is the sequential implementation of K-Means algorithm. The second part is on how to applied the froms of parallelism, including shared memory using OpenMp and NVIDIA’s Graphics Processing
Units with CUDA.

To start with, the three major challenges in implementing K-Means are initializing centroids, determining number of centroids, and determining number of iterations. We have to select proper intial centroids to avoid that the algorithm converges to a local minimum. Moreover, the program requires iterations and clusters as inputs. Thus, it is difficult to predict the best values as these can vary with different input datasets.

For parallelism, the major challenge is achieve a high speedup even when workload is imbalanced. We have to figure out a way to maintain the quality of the K-Means algorithm's results and keep a resonable convergence rate in parallelization. Additionally, we need to implement a shared memory model with a multi-thread program to spawn parallel computation for OpenMp approach, and a heterogeneous host-device implementation with CPU and GPU for CUDA approach. Performing required calculations to build index clustering array on GPU can be challenging as well.

### RESOURCES
The project idea is inspired by a research paper Speeding up k-Means algorithm by GPUs (https://www.sciencedirect.com/science/article/pii/S0022000012000992).
We will start with implementin a sequential verision of K-Means from stratch.
The testing datasets will be obtained from Kaggle (will be added as project goes).
We would refer to research papers in related field and the tutorials on how to run our parallelized version of the algorithm.

### GOALS AND DELIVERABLES
Plan to achieve:
* Implement a sequential verision of K-Means algorithm as a baseline testbench.
* Parallel implementations of the K-Means clustering algorithm using OpenMp
* Parallel implementations of the K-Means clustering algorithm using CUDA
* Perform an evaluation of the speedup and efficiency acheived in the above implementations.

Hope to achieve:
* Try to implement K-Means clustering algorithm on other parallel models like MPI, and comparing these frameworks with the above versions.

### PLATFORM CHOICE
We will implement  K-Means clustering algorithm in C/CUDA on NVIDIA 1080 GPU on GHC machines. Other parallel models will be implemented and tested using GHC machines as well.

### SCHEDULE
* 03/21 - 03/27: Formulate input datasets by using Kaggle platform. Setup sequential K-Means algorithm code skeleton.
* 03/28 - 04/03: Choose the method to intialize K-Means centroids, determine clusters and iterations. Implement a sequential K-Means algorithm using these methods.
* 04/04 - 04/10: Parallel the K-Means algorithm with CUDA. Obtain some results by the intermediate checkpoint of the speedups from the parallel version.
* 04/11 - 04/17: Parallel the K-Means algorithm with OpenMp. 
* 04/18- 04/24: Optimize the models and perform code cleanup.
* 04/25 - 04/29: Put together the report and explore other frameworks if time is allowed.
