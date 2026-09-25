import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

class CandidatePairGenerator:
    def __init__(self, model_save_path: str, total_threads: int = 32, python_threads: int = 8):
        """
        Initializes the CandidatePairGenerator with the specified SentenceTransformer model and threading configuration.

        Parameters:
        - model_save_path (str): Path to the pre-trained SentenceTransformer model.
        - total_threads (int): Total number of CPU threads available.
        - python_threads (int): Number of threads to allocate for Python operations.
        """
        self.model_save_path = model_save_path
        self.model = SentenceTransformer(self.model_save_path)
        
        # Thread configuration
        self.TOTAL_THREADS = total_threads
        self.PYTHON_THREADS = python_threads
        self.FAISS_THREADS = self.TOTAL_THREADS - self.PYTHON_THREADS
        faiss.omp_set_num_threads(self.FAISS_THREADS)
        
    @staticmethod
    def serialize_record(record: pd.Series, columns: list) -> str:
        """
        Serializes a single record into a string

        Parameters:
        - record (pd.Series): A row from the dataframe.
        - columns (list): List of column names to include in serialization.

        Returns:
        - str: Serialized string representation of the record.
        """
        serialized_str = ""
        for col_name in columns:
            value = record[col_name]
            serialized_str += f" {value}"
        return serialized_str.strip().lower()
    
    def generate_embeddings(self, df: pd.DataFrame, id_col: str, serialize_columns: list) -> pd.DataFrame:
        """
        Serializes records and generates L2-normalized embeddings.

        Parameters:
        - df (pd.DataFrame): Input dataframe containing records.
        - id_col (str): Column name representing the unique identifier.
        - serialize_columns (list): List of columns to include in serialization.

        Returns:
        - pd.DataFrame: DataFrame containing IDs and their corresponding normalized embeddings.
        """
        print(f"Serializing records for {id_col}...")
        serialized = df.apply(
            lambda row: self.serialize_record(row, serialize_columns), axis=1
        ).tolist()
        
        print(f"Generating embeddings for {id_col}...")
        embeddings = self.model.encode(
            serialized,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=True
        )
        
        print(f"Normalizing embeddings for {id_col}...")
        embeddings_normalized = normalize(embeddings, norm='l2', axis=1)
        
        # Ensure that embeddings are stored as lists to preserve their dimensions
        embeddings_df = pd.DataFrame({
            id_col: df[id_col],
            "embedding": list(embeddings_normalized)
        })
        
        print(f"{id_col} Embeddings shape: {embeddings_df.shape}")
        return embeddings_df
    
    @staticmethod
    def compute_norms(embeddings: np.ndarray, name: str):
        """
        Computes and prints the norms of embeddings.

        Parameters:
        - embeddings (np.ndarray): Embedding vectors.
        - name (str): Name identifier for the embeddings.
        """
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"{name} Embeddings Norms:")
        print(f"First 5 norms: {norms[:5]}")
        print(f"Min norm: {norms.min()}, Max norm: {norms.max()}\n")
        return norms
    
    def verify_and_normalize(self, embeddings: np.ndarray, name: str) -> np.ndarray:
        """
        Verifies that embeddings are L2-normalized and re-normalizes if necessary.

        Parameters:
        - embeddings (np.ndarray): Embedding vectors.
        - name (str): Name identifier for the embeddings.

        Returns:
        - np.ndarray: Verified and normalized embeddings.
        """
        norms = np.linalg.norm(embeddings, axis=1)
        min_norm = norms.min()
        max_norm = norms.max()
        print(f"{name} - Norms: Min = {min_norm:.4f}, Max = {max_norm:.4f}")
        
        if not np.allclose(norms, 1.0, atol=1e-4):
            print(f"Re-normalizing {name} embeddings...")
            embeddings = normalize(embeddings, norm='l2', axis=1)
            norms = np.linalg.norm(embeddings, axis=1)
            print(f"After Re-normalization - {name} - Norms: Min = {norms.min():.4f}, Max = {norms.max():.4f}")
        else:
            print(f"{name} embeddings are already L2-normalized.")
        
        return embeddings
    
    @staticmethod
    def get_batches(data: np.ndarray, batch_size: int):
        """
        Generator that yields batches of data.

        Parameters:
        - data (np.ndarray): The dataset to be divided into batches.
        - batch_size (int): The number of samples per batch.

        Yields:
        - np.ndarray: A batch of data.
        """
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def perform_knn_search(self, index: faiss.Index, batch_queries: np.ndarray, k: int, 
                           df1_ids_batch: np.ndarray, df2_ids: np.ndarray, 
                           similarity_threshold: float):
        """
        Performs a k-NN search on a batch of queries and filters based on similarity threshold.

        Parameters:
        - index (faiss.Index): The Faiss index to search against.
        - batch_queries (np.ndarray): The batch of query embeddings.
        - k (int): The number of nearest neighbors to retrieve.
        - df1_ids_batch (np.ndarray): The IDs corresponding to the batch queries.
        - df2_ids (np.ndarray): The IDs for mapping.
        - similarity_threshold (float): The minimum similarity score to consider.

        Returns:
        - tuple of lists: (list_df1_ids, list_df2_ids, list_similarities)
        """
        list_df1_ids = []
        list_df2_ids = []
        list_similarities = []
        
        # Perform k-NN search
        D, I = index.search(batch_queries, k)
        
        for i in range(len(batch_queries)):
            df1_id = df1_ids_batch[i]
            for j in range(k):
                similarity = D[i][j]
                df2_idx = I[i][j]
                if similarity >= similarity_threshold:
                    df2_id = df2_ids[df2_idx]
                    list_df1_ids.append(df1_id)
                    list_df2_ids.append(df2_id)
                    list_similarities.append(similarity)
                    
        return (list_df1_ids, list_df2_ids, list_similarities)
    
    def generate_candidate_pairs(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df1_id_col: str,
        df2_id_col: str,
        serialize_columns: list,
        similarity_threshold: float = 0.70,
        k: int = 10,
        batch_size_per_thread: int = 32
    ) -> pd.DataFrame:
        """
        Generates candidate pairs based on cosine similarity using k-NN search.

        Parameters:
        - df1 (pd.DataFrame): First dataframe containing records.
        - df2 (pd.DataFrame): Second dataframe containing records.
        - df1_id_col (str): Column name for the first dataframe's unique identifier.
        - df2_id_col (str): Column name for the second dataframe's unique identifier.
        - serialize_columns (list): List of columns to include in serialization (assumed to be the same for both dataframes).
        - similarity_threshold (float): Minimum cosine similarity to consider.
        - k (int): Number of nearest neighbors to retrieve.
        - batch_size_per_thread (int): Number of queries per batch per thread.

        Returns:
        - pd.DataFrame: DataFrame containing candidate pairs with df1_id, df2_id, and similarity score.
        """
        # Generate embeddings for df1 and df2
        df1_embeddings_df = self.generate_embeddings(df1, df1_id_col, serialize_columns)
        df2_embeddings_df = self.generate_embeddings(df2, df2_id_col, serialize_columns)
        
        # Extract embeddings as float32 numpy arrays
        df1_embeddings_float = np.vstack(df1_embeddings_df['embedding'].values).astype('float32')
        df2_embeddings_float = np.vstack(df2_embeddings_df['embedding'].values).astype('float32')
        
        # Extract IDs
        df1_ids = df1_embeddings_df[df1_id_col].values
        df2_ids = df2_embeddings_df[df2_id_col].values
        
        print(f"{df1_id_col} Embeddings shape: {df1_embeddings_float.shape}")
        print(f"{df2_id_col} Embeddings shape: {df2_embeddings_float.shape}")
        
        # Verify and normalize embeddings
        df1_embeddings_float = self.verify_and_normalize(df1_embeddings_float, f"{df1_id_col} Embeddings")
        df2_embeddings_float = self.verify_and_normalize(df2_embeddings_float, f"{df2_id_col} Embeddings")
        
        # Verify embeddings data types
        if df1_embeddings_float.dtype != np.float32:
            warnings.warn(f"{df1_id_col} embeddings are not float32. Converting...")
            df1_embeddings_float = df1_embeddings_float.astype('float32')
        
        if df2_embeddings_float.dtype != np.float32:
            warnings.warn(f"{df2_id_col} embeddings are not float32. Converting...")
            df2_embeddings_float = df2_embeddings_float.astype('float32')
        
        # Initialize Faiss index for Inner Product (cosine similarity)
        dimension = df1_embeddings_float.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Exact search
        index.add(df2_embeddings_float)
        print(f"Faiss index initialized with {index.ntotal} vectors.")
        
        # Configure threading and batching
        BATCH_SIZE_PER_THREAD = batch_size_per_thread
        TOTAL_BATCH_SIZE = BATCH_SIZE_PER_THREAD * self.PYTHON_THREADS
        
        # Initialize lists to store candidate pairs
        list_df1_ids = []
        list_df2_ids = []
        list_similarities = []
        
        print("\nPerforming parallel k-NN searches...")
        with ThreadPoolExecutor(max_workers=self.PYTHON_THREADS) as executor:
            future_to_batch = {}
            for batch_queries, batch_ids in zip(
                self.get_batches(df1_embeddings_float, TOTAL_BATCH_SIZE),
                self.get_batches(df1_ids, TOTAL_BATCH_SIZE)
            ):
                # Further split into sub-batches per thread
                for sub_batch_queries, sub_batch_ids in zip(
                    self.get_batches(batch_queries, BATCH_SIZE_PER_THREAD),
                    self.get_batches(batch_ids, BATCH_SIZE_PER_THREAD)
                ):
                    future = executor.submit(
                        self.perform_knn_search,
                        index,
                        sub_batch_queries,
                        k,
                        sub_batch_ids,
                        df2_ids,
                        similarity_threshold
                    )
                    future_to_batch[future] = (sub_batch_queries, sub_batch_ids)
        
            # Collect results as they complete
            #for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Processing Batches"):
            for future in as_completed(future_to_batch):
                try:
                    batch_df1_ids, batch_df2_ids, batch_similarities = future.result()
                    list_df1_ids.extend(batch_df1_ids)
                    list_df2_ids.extend(batch_df2_ids)
                    list_similarities.extend(batch_similarities)
                except Exception as exc:
                    print(f"Batch generated an exception: {exc}")
        
        print("Parallel k-NN searches completed.")
        
        # Create DataFrame of candidate pairs
        candidate_pairs_df = pd.DataFrame({
            df1_id_col: list_df1_ids,
            df2_id_col: list_df2_ids,
            "similarity": list_similarities
        })
        
        print(f"\nTotal candidate pairs before deduplication: {len(candidate_pairs_df)}")
        
        # Remove duplicate pairs
        candidate_pairs_df = candidate_pairs_df.drop_duplicates(subset=[df1_id_col, df2_id_col]).reset_index(drop=True)
        print(f"Total candidate pairs after deduplication: {len(candidate_pairs_df)}")
        
        # Verify similarity scores
        min_similarity = candidate_pairs_df['similarity'].min()
        max_similarity = candidate_pairs_df['similarity'].max()
        print(f"\nSimilarity scores range: Min = {min_similarity:.4f}, Max = {max_similarity:.4f}")
        
        return candidate_pairs_df
