import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
class LoadData:
    def __init__(self):
        """
        Initialize the connection to the PostgreSQL database.
        """
        self.host = os.getenv("DB_HOST")
        self.database= os.getenv("DB_NAME")
        self.port= os.getenv("DB_PORT")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.connection = None
    
    def connect(self):
        """
        Establish a connection to the PostgreSQL database.
        """
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
        except Exception as e:
            print(f"Error connecting to PostgreSQL database: {e}")
    
    def fetch_data(self, query, params=None):
        """
        Fetch data from the PostgreSQL database and return as a Pandas DataFrame.
        
        Parameters:
        query (str): The SQL query to execute.
        params (tuple): Optional tuple of parameters to pass with the query.
        
        Returns:
        pd.DataFrame: A DataFrame containing the result set.
        """
        if self.connection is None:
            print("No connection established.")
            return None
        
        try:
            # Use pandas to execute the query and return a DataFrame
            df = pd.read_sql(query, self.connection, params=params)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    
    def execute_query(self, query, params=None):
        """
        Execute a write operation (INSERT, UPDATE, DELETE).
        """
        if self.connection is None:
            print("No connection established.")
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error executing query: {e}")
            self.connection.rollback()  # Rollback in case of error
            return False
    
    def close(self):
        """
        Close the connection to the PostgreSQL database.
        """
        if self.connection is not None:
            self.connection.close()
            print("Connection closed.")
