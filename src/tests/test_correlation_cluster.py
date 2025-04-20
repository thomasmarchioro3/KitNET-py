import logging
import time
import unittest

# External libraries
from sklearn.datasets import make_classification

# Local modules
from src.correlation_cluster import CorrelationCluster
from original.corClust import corClust


logger = logging.Logger("test_correlation_cluster")


class TestCorrelationCluster(unittest.TestCase):

    def setUp(self):

        X, _ = make_classification(
            n_samples=10_000, 
            n_features=20, 
            n_informative=10
        )

        self.X = X
        self.num_features = X.shape[1]
        logger.addHandler(logging.StreamHandler())
        return super().setUp()

    def test_correlation_cluster(self):

        max_cluster_size = 10
        
        tic = time.perf_counter()
        corr_cluster_original = corClust(self.num_features)
        for x in self.X:
            corr_cluster_original.update(x)
        clusters_orig = corr_cluster_original.cluster(max_cluster_size)
        time_orig = time.perf_counter() - tic 

        tic = time.perf_counter()
        corr_cluster = CorrelationCluster(self.num_features)
        for x in self.X:
            corr_cluster.update(x)
        clusters = corr_cluster.get_clusters(max_cluster_size)
        time_refactored = time.perf_counter() - tic

        for cluster, cluster_orig in zip(clusters, clusters_orig):
            logger.log(level=logging.DEBUG, msg=str(cluster))
            self.assertEqual(set(cluster), set(cluster_orig))

        logger.log(level=logging.DEBUG, msg=f"Correlation cluster (original): {time_orig:.4f}")
        logger.log(level=logging.DEBUG, msg=f"Correlation cluster (refactored): {time_refactored:.4f}")


if __name__ == "__main__":

    log_level = logging.DEBUG
    logger.setLevel(level=log_level)
    unittest.main()
        