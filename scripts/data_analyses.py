import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalysis:
    def __init__(self, df):
        """
        Initialize with a DataFrame containing telecom data.
        
        Parameters:
        df (pd.DataFrame): The session data.
        """
        self.df = df
    
    def describe_variables(self):
        """
        Describe all relevant variables and their data types.
        """
        description = self.df.describe(include='all')
        data_types = self.df.dtypes
        return description, data_types

    def segment_users_by_duration(self):
        """
        Segment users into deciles based on total session duration, compute the total data (DL + UL) per decile.
        
        Returns:
        pd.DataFrame: Total data per decile class.
        """
        # Create a column for total session duration per user
        self.df['total_duration'] = self.df.groupby('IMSI')['Dur. (ms)'].transform('sum')
        
        # Segment users into deciles based on the total duration
        self.df['decile'] = pd.qcut(self.df['total_duration'], 10, labels=False)
        
        # Compute total data (DL + UL) per decile
        self.df['total_data'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']
        
        # Aggregate total data by decile
        decile_data = self.df.groupby('decile')['total_data'].sum().reset_index()
        return decile_data

    def basic_metrics(self):
        """
        Compute basic metrics (mean, median, etc.) and their importance.
        """
        mean = self.df.mean()
        median = self.df.median()
        std_dev = self.df.std()
        min_val = self.df.min()
        max_val = self.df.max()

        return pd.DataFrame({'mean': mean, 'median': median, 'std_dev': std_dev, 'min': min_val, 'max': max_val})

    def non_graphical_univariate_analysis(self):
        """
        Compute dispersion parameters for each quantitative variable (variance, std, range).
        """
        dispersion = {
            'variance': self.df.var(),
            'std_dev': self.df.std(),
            'range': self.df.max() - self.df.min()
        }
        return pd.DataFrame(dispersion)

    def graphical_univariate_analysis(self):
        """
        Perform graphical univariate analysis using histograms and box plots.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Plot histograms for each numeric column
        for col in numeric_cols:
            plt.figure(figsize=(10, 5))
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f'Histogram for {col}')
            plt.show()
        
        # Plot boxplots for each numeric column
        for col in numeric_cols:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=self.df[col].dropna())
            plt.title(f'Boxplot for {col}')
            plt.show()

    def bivariate_analysis(self):
        """
        Explore the relationship between each application and total data (DL + UL).
        """
        # Add total data for each application
        app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                       'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)',
                       'Social Media UL (Bytes)', 'Google UL (Bytes)', 'Email UL (Bytes)',
                       'Youtube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)', 'Other UL (Bytes)']
        self.df['total_app_data'] = self.df[app_columns].sum(axis=1)

        # Compute correlation between applications and total data
        app_data = self.df[['total_data'] + app_columns]
        corr_matrix = app_data.corr()

        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation between Applications and Total Data')
        plt.show()

        return corr_matrix

    def correlation_analysis(self):
        """
        Compute a correlation matrix for selected application data.
        """
        # Application columns for correlation analysis
        app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                       'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        
        # Select only the relevant columns
        app_data = self.df[app_columns]
        
        # Compute the correlation matrix
        corr_matrix = app_data.corr()

        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title('Correlation Matrix for Application Data')
        plt.show()
        
        return corr_matrix

    def perform_pca(self, n_components=2):
        """
        Perform Principal Component Analysis (PCA) to reduce dimensionality.
        
        Parameters:
        n_components (int): Number of principal components to compute.
        
        Returns:
        pd.DataFrame: PCA components.
        """
        # Select numerical columns for PCA
        app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                       'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        app_data = self.df[app_columns].dropna()

        # Normalize the data before PCA
        app_data_normalized = (app_data - app_data.mean()) / app_data.std()

        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(app_data_normalized)

        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        
        # Return the PCA components and explained variance
        return pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(n_components)]), explained_variance

