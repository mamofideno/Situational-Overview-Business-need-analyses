import pandas as pd

class DataAggregator:
    def __init__(self, session_data):
        """
        Initialize the TelecomDataAggregator class with session data.

        Parameters:
        session_data (pd.DataFrame): The session data containing information on telecom sessions.
        Expected columns include 'IMSI', 'MSISDN/Number', 'Bearer Id', 'Dur. (ms)', 'Total DL (Bytes)',
        'Total UL (Bytes)', and various application-level columns like 'Social Media DL (Bytes)', etc.
        """
        self.session_data = session_data

    def aggregate_per_user(self, user_col='IMSI'):
        """
        Aggregates the data per user (IMSI or MSISDN).

        Parameters:
        user_col (str): The column representing the user, either 'IMSI' or 'MSISDN/Number'.
        
        Returns:
        pd.DataFrame: Aggregated data per user including:
        - Number of xDR sessions
        - Total session duration
        - Total download (DL) and upload (UL) data
        - Total data volume per application for each user
        """
        # List of application columns
        app_columns_dl = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                          'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        app_columns_ul = ['Social Media UL (Bytes)', 'Google UL (Bytes)', 'Email UL (Bytes)', 
                          'Youtube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)', 'Other UL (Bytes)']
        
        # Aggregate number of sessions, total session duration, and total data volume (DL/UL)
        user_aggregates = self.session_data.groupby(user_col).agg(
            number_of_sessions=('Bearer Id', 'nunique'),       # Count unique sessions
            total_session_duration=('Dur. (ms)', 'sum'),       # Sum session duration
            total_download=('Total DL (Bytes)', 'sum'),        # Sum total download data
            total_upload=('Total UL (Bytes)', 'sum')           # Sum total upload data
        ).reset_index()

        # Aggregate total data volume for each application (DL and UL)
        app_aggregates_dl = self.session_data.groupby(user_col)[app_columns_dl].sum().reset_index()
        app_aggregates_ul = self.session_data.groupby(user_col)[app_columns_ul].sum().reset_index()

        # Merge the user-level aggregates with the application-level aggregates (DL and UL)
        result = user_aggregates.merge(app_aggregates_dl, on=user_col, how='left')
        result = result.merge(app_aggregates_ul, on=user_col, how='left')

        return result
