#include <fstream>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <armadillo>

//Library dependencies for linking -larmadillo -lmlpack -lboost_serialization  

int main() {
    uint8_t k = 2; 
    uint8_t dim = 2; 
    uint8_t samples = 50;
    uint8_t iterations = 10;     
    arma::mat data(dim, samples, arma::fill::zeros);

    // Initialize data    
    for(int32_t i = 0; i < samples/2; ++i)
        data.col(i) = arma::vec({1, 1}) + 0.25*arma::randn<arma::vec>(dim);
 
    for(; i <samples; ++i)
        data.col(i) = arma::vec({2, 3}) + 0.25*arma::randn<arma::vec>(dim);

    // Cluster the data
    arma::Row<size_t> clusters;
    arma::mat centroids;
    mlpack::kmeans::KMeans<> mlpack_kmeans(iterations);
    mlpack_kmeans.Cluster(data, k, clusters, centroids);

    centroids.print("Centroids:");
    return 0;
