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

## Milestone 
### Updates
We implemented a sequential version of a K-means clustering algorithm which groups data points into a number of clusters. The data points are defined as points_t structure, which holds a point's coordinate values, cluster id, and its minimum distance to a cluster. The clusters are the input parameter set by the user. We also use a centroids_t structure array to track the updated centroids in each iteration. The algorithm is made up of three parts, the initialization stage, update stage, and convergence stage. In the initialization section, we initialize the centroid array by randomly selecting N points. In the update stage, we recompute the centroid of each cluster using Lloyd’s algorithm. Then, we assign each point to the nearest centroid and redefine the cluster. In the convergence stage, we conclude the computation is complete if the centroid of each cluster is the same for two continuous iterations.

We also created a benchmark for the K-means clustering algorithm. The datasets are in the data folder. It contains datasets with different sizes, clusters, and skewness. In addition, we set up the timer for initialization and computation.

We also began implementing a parallel version of this code in CUDA, but unfortunately, there are still some bugs that exist in the implementation that has to be fixed. For this reason, I do not have any measurements for that. 

### New Schedule
* 04/11 - 04/14: Parallel the K-Means algorithm with CUDA, Fix bugs. 
* 04/15 - 04/18: Parallel the K-Means algorithm with OpenMp, figure out and implement ways to parallel K-means clustering algorithm. 
* 04/18- 04/21: Optimize the models and perform code cleanup.
* 04/22- 04/25: Explore variations on the sequential K-means algorithm using different centroid selection algorithms and convergence tests.
* 04/26 - 04/29: Put together the report and explore other frameworks if time is allowed. [Plan to show graph at the poster session]

### Goals
* Produce graphs which would help convey how much parallelism can be achieved with different approaches to build K-means clustering algorithm
* Parallelize an application of k-means algorithm and show speedups.
* Discuss if there are any tradeoffs between speedup and clustering accuracy.
* If we have time, we would compare the speedup betweern CUDA, OpenMp, and other framworks.

### Peliminary Results
Currently, I implemented a visualization and timing function of the sequential K-means clustering algorithm. Taking the dataset data/3000_20_a1.txt as an example. below are its clustering and timing results.

![figure_1](https://user-images.githubusercontent.com/43794945/162873360-46d776fb-42ca-440f-ad2f-80386f5131f6.png)
![image](https://user-images.githubusercontent.com/43794945/162873566-0738af29-1166-4180-905d-29e30fac19c7.png)

