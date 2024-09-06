import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class CustomerEngagement:
    def __init__(self, df, user_col='MSISDN/Number'):
        """
        Initialize the TelecomCustomerEngagement class with session data.

        Parameters:
        df (pd.DataFrame): The session data containing information on telecom sessions.
        user_col (str): The column representing the user, e.g., 'MSISDN/Number'.
        """
        self.df = df.copy()
        self.user_col = user_col
        self.aggregated_df = None
        self.top_customers = {}
        self.scaled_metrics = None
        self.kmeans_model = None
        self.clustered_df = None
        self.cluster_stats = None
        self.top_users_per_app = {}
        self.pca_components = None

    def aggregate_metrics_per_customer(self):
        """
        Aggregate metrics per customer (MSISDN).

        Metrics:
        - Number of xDR sessions
        - Total session duration
        - Total DL and UL data
        - Total data volume per application

        Returns:
        pd.DataFrame: Aggregated metrics per customer.
        """
        # Define application names based on data structure
        # applications for DL and UL are: 'Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other'
        application_names = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

        # Create total data per application columns
        for app in application_names:
            dl_col = f'{app} DL (Bytes)'
            ul_col = f'{app} UL (Bytes)'
            total_col = f'{app} Total (Bytes)'
            if dl_col in self.df.columns and ul_col in self.df.columns:
                self.df[total_col] = self.df[dl_col] + self.df[ul_col]
            else:
                # Handle missing columns by creating them with 0
                self.df[dl_col] = self.df.get(dl_col, 0)
                self.df[ul_col] = self.df.get(ul_col, 0)
                self.df[total_col] = self.df[dl_col] + self.df[ul_col]

        # Define the aggregation dictionary
        agg_dict = {
            'Bearer Id': 'nunique',  
            'Dur. (ms)': 'sum',      # total_session_duration
            'Total DL (Bytes)': 'sum',  # total_download
            'Total UL (Bytes)': 'sum'   # total_upload
        }

        # Add application total columns to agg_dict
        for app in application_names:
            total_col = f'{app} Total (Bytes)'
            agg_dict[total_col] = 'sum'

        # Perform aggregation
        aggregated = self.df.groupby(self.user_col).agg(agg_dict).reset_index()

        # Rename columns to meaningful names
        aggregated = aggregated.rename(columns={
            'Bearer Id': 'number_of_sessions',
            'Dur. (ms)': 'total_session_duration',
            'Total DL (Bytes)': 'total_download',
            'Total UL (Bytes)': 'total_upload'
        })

        # Assign to instance variable
        self.aggregated_df = aggregated

        return self.aggregated_df

    def report_top_10_customers_per_metric(self, metrics):
        """
        Report the top 10 customers per engagement metric.

        Parameters:
        metrics (list of str): List of metric column names to report top 10 for.

        Returns:
        dict: Dictionary where keys are metrics and values are DataFrames of top 10 customers.
        """
        if self.aggregated_df is None:
            raise ValueError("Aggregated data not found. Run aggregate_metrics_per_customer first.")

        top_customers = {}
        for metric in metrics:
            if metric not in self.aggregated_df.columns:
                print(f"Metric '{metric}' not found in aggregated data.")
                continue
            top10 = self.aggregated_df.nlargest(10, metric)
            top_customers[metric] = top10
        self.top_customers = top_customers
        return top_customers

    def normalize_metrics(self, metrics):
        """
        Normalize the specified engagement metrics using StandardScaler.

        Parameters:
        metrics (list of str): List of metric column names to normalize.

        Returns:
        pd.DataFrame: Scaled metrics.
        """
        if self.aggregated_df is None:
            raise ValueError("Aggregated data not found. Run aggregate_metrics_per_customer first.")

        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.aggregated_df[metrics])
        scaled_df = pd.DataFrame(scaled, columns=[f'scaled_{col}' for col in metrics])
        self.scaled_metrics = scaled_df
        return scaled_df

    def find_optimal_k_elbow_method(self, max_k=10):
        """
        Find the optimal k using the elbow method by plotting the inertia for different k values.

        Parameters:
        max_k (int): Maximum number of clusters to test.

        Returns:
        None: Displays the elbow plot.
        """
        if self.scaled_metrics is None:
            raise ValueError("Scaled metrics not found. Run normalize_metrics first.")

        inertia = []
        K = range(1, max_k + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.scaled_metrics)
            inertia.append(kmeans.inertia_)

        # Plot the elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.xticks(K)
        plt.show()

        # Note: User needs to inspect the plot to determine the optimal k.

    def run_kmeans(self, k=3):
        """
        Run K-Means clustering on normalized metrics.

        Parameters:
        k (int): Number of clusters.

        Returns:
        KMeans: Trained KMeans model.
        """
        if self.scaled_metrics is None:
            raise ValueError("Scaled metrics not found. Run normalize_metrics first.")

        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(self.scaled_metrics)
        self.kmeans_model = kmeans

        # Assign cluster labels to the aggregated data
        self.clustered_df = self.aggregated_df.copy()
        self.clustered_df['cluster'] = kmeans.labels_

        return kmeans

    def compute_cluster_statistics(self):
        """
        Compute the min, max, average, and total non-normalized metrics for each cluster.

        Returns:
        pd.DataFrame: Cluster statistics.
        """
        if self.clustered_df is None:
            raise ValueError("Clustered data not found. Run run_kmeans first.")

        # Define non-normalized metrics
        metrics = ['number_of_sessions', 'total_session_duration', 'total_download', 'total_upload']
        # Identify application total columns
        application_metrics = [col for col in self.aggregated_df.columns if 'Total (Bytes)' in col]
        all_metrics = metrics + application_metrics

        # Compute statistics
        cluster_stats = self.clustered_df.groupby('cluster').agg({
            'number_of_sessions': ['min', 'max', 'mean', 'sum'],
            'total_session_duration': ['min', 'max', 'mean', 'sum'],
            'total_download': ['min', 'max', 'mean', 'sum'],
            'total_upload': ['min', 'max', 'mean', 'sum']
        })

        # Add application metrics statistics
        for app_col in application_metrics:
            cluster_stats[(app_col, 'min')] = self.clustered_df.groupby('cluster')[app_col].min()
            cluster_stats[(app_col, 'max')] = self.clustered_df.groupby('cluster')[app_col].max()
            cluster_stats[(app_col, 'mean')] = self.clustered_df.groupby('cluster')[app_col].mean()
            cluster_stats[(app_col, 'sum')] = self.clustered_df.groupby('cluster')[app_col].sum()

        # Flatten MultiIndex columns
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        cluster_stats = cluster_stats.reset_index()

        self.cluster_stats = cluster_stats
        return cluster_stats

    def visualize_cluster_statistics(self):
        """
        Visualize cluster statistics using bar plots for mean values.

        Displays:
        - Number of Sessions
        - Total Session Duration
        - Total Download
        - Total Upload
        """
        if self.cluster_stats is None:
            raise ValueError("Cluster statistics not found. Run compute_cluster_statistics first.")

        # Define metrics to plot
        metrics = ['number_of_sessions', 'total_session_duration', 'total_download', 'total_upload']

        # Create a subplot for each metric
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cluster Mean Statistics', fontsize=16)

        for ax, metric in zip(axs.flatten(), metrics):
            metric_mean_col = f'{metric}_mean'
            if metric_mean_col in self.cluster_stats.columns:
                sns.barplot(x='cluster', y=metric_mean_col, data=self.cluster_stats, ax=ax, palette='Set2')
                ax.set_title(f'Mean {metric.replace("_", " ").title()}')
                ax.set_xlabel('Cluster')
                ax.set_ylabel(f'Mean {metric.replace("_", " ").title()}')
            else:
                ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def visualize_clusters_with_pca(self):
        """
        Visualize clusters by reducing data to 2D using PCA and plotting.

        Displays:
        - Scatter plot of first two principal components colored by cluster.
        """
        if self.clustered_df is None:
            raise ValueError("Clustered data not found. Run run_kmeans first.")
        if self.scaled_metrics is None:
            raise ValueError("Scaled metrics not found. Run normalize_metrics first.")

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.scaled_metrics)
        principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        principal_df['cluster'] = self.clustered_df['cluster']

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=principal_df, palette='Set1', alpha=0.7)
        plt.title('Clusters Visualization using PCA Components')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.show()

    def aggregate_top_users_per_application(self, top_n=10):
        """
        Aggregate user total traffic per application and derive the top n most engaged users per application.

        Parameters:
        top_n (int): Number of top users to retrieve per application.

        Returns:
        dict: Dictionary where keys are applications and values are DataFrames of top n users.
        """
        if self.aggregated_df is None:
            raise ValueError("Aggregated data not found. Run aggregate_metrics_per_customer first.")

        # Identify application total columns
        application_metrics = [col for col in self.aggregated_df.columns if 'Total (Bytes)' in col]
        top_users_per_app = {}

        for app_col in application_metrics:
            app_name = app_col.replace(' Total (Bytes)', '')
            top_users = self.aggregated_df.nlargest(top_n, app_col)[[self.user_col, app_col]].reset_index(drop=True)
            top_users_per_app[app_name] = top_users

        self.top_users_per_app = top_users_per_app
        return top_users_per_app

    def plot_top_applications(self, top_n=3):
        """
        Plot the top n most used applications based on total data usage.

        Parameters:
        top_n (int): Number of top applications to plot.

        Displays:
        - Bar plot of top n applications by total data usage.
        """
        if self.aggregated_df is None:
            raise ValueError("Aggregated data not found. Run aggregate_metrics_per_customer first.")

        # Identify application total columns
        application_metrics = [col for col in self.aggregated_df.columns if 'Total (Bytes)' in col]
        # Sum total data for each application
        app_totals = self.aggregated_df[application_metrics].sum().sort_values(ascending=False)
        # Select top_n applications
        top_apps = app_totals.head(top_n)
        top_apps_df = top_apps.reset_index()
        top_apps_df.columns = ['Application', 'Total Bytes']

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Total Bytes', y='Application', data=top_apps_df, palette='viridis')
        plt.title(f'Top {top_n} Most Used Applications')
        plt.xlabel('Total Data (Bytes)')
        plt.ylabel('Application')
        plt.show()

    def run_kmeans_with_elbow(self, k=3, max_k=10):
        """
        Find the optimal k using the elbow method and run K-Means clustering.

        Parameters:
        k (int): Number of clusters for K-Means.
        max_k (int): Maximum number of clusters to test with elbow method.

        Returns:
        KMeans: Trained KMeans model.
        """
        # Step 1: Find the optimal k using elbow method
        self.find_optimal_k_elbow_method(max_k=max_k)

        # Step 2: Run K-Means with specified k
        kmeans = self.run_kmeans(k=k)

        # Step 3: Compute cluster statistics
        self.compute_cluster_statistics()

        # Step 4: Visualize cluster statistics
        self.visualize_cluster_statistics()
        self.visualize_clusters_with_pca()

        return kmeans

    def get_top_customers(self):
        """
        Get the top 10 customers per engagement metric.

        Returns:
        dict: Dictionary where keys are metrics and values are DataFrames of top 10 customers.
        """
        return self.top_customers

    def get_clustered_data(self):
        """
        Get the clustered data with cluster labels.

        Returns:
        pd.DataFrame: Clustered data.
        """
        return self.clustered_df

    def get_cluster_statistics(self):
        """
        Get the cluster statistics.

        Returns:
        pd.DataFrame: Cluster statistics.
        """
        return self.cluster_stats

    def get_top_users_per_application(self):
        """
        Get the top users per application.

        Returns:
        dict: Dictionary with application names as keys and DataFrames of top users as values.
        """
        return self.top_users_per_app

    def interpret_cluster_statistics(self):
        """
        Interpret the cluster statistics.

        Returns:
        None: Prints interpretation of cluster statistics.
        """
        if self.cluster_stats is None:
            raise ValueError("Cluster statistics not found. Run compute_cluster_statistics first.")

        print("Cluster Statistics Interpretation:")
        print("-" * 50)
        for index, row in self.cluster_stats.iterrows():
            cluster = row['cluster']
            print(f"Cluster {cluster}:")
            print(f"  Number of Sessions: Min={row['number_of_sessions_min']}, Max={row['number_of_sessions_max']}, Mean={row['number_of_sessions_mean']:.2f}, Sum={row['number_of_sessions_sum']}")
            print(f"  Total Session Duration (ms): Min={row['total_session_duration_min']}, Max={row['total_session_duration_max']}, Mean={row['total_session_duration_mean']:.2f}, Sum={row['total_session_duration_sum']}")
            print(f"  Total Download (Bytes): Min={row['total_download_min']}, Max={row['total_download_max']}, Mean={row['total_download_mean']:.2f}, Sum={row['total_download_sum']}")
            print(f"  Total Upload (Bytes): Min={row['total_upload_min']}, Max={row['total_upload_max']}, Mean={row['total_upload_mean']:.2f}, Sum={row['total_upload_sum']}")
            print("\n")
            # For applications, similar interpretation can be added if needed.
