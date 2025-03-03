import streamlit as st 
import sqlite3
import pandas as pd
import os

if 'file_uploaded' not in st.session_state:
    st.session_state['file_uploaded'] = False

def detect_sqlite_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "REAL"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    else:
        return "TEXT"

def main():
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.title("Dataset Uploader")
    
    if not st.session_state.file_uploaded:
        st.session_state.uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if st.session_state.uploaded_file is not None:
            if st.button("Process"):
                st.session_state['file_uploaded'] = True
                st.rerun()
    
    if st.session_state.get('file_uploaded'):
        try:
            if 'data' not in st.session_state:
                st.session_state.data = pd.read_csv(st.session_state.uploaded_file)
            columns = st.session_state.data.columns.tolist()
            st.write(st.session_state.data.head())
            selected = st.selectbox("Select the column to be used as class label", columns)
            name_table = st.text_input(label="Insert the name of the table", value=st.session_state.uploaded_file.name.split(".")[0])
            if st.button("Process"):
                conn = sqlite3.connect('medinet.db')
                cursor = conn.cursor()
                
                # Create table for dataset with detected types
                column_types = [f"{col} {detect_sqlite_type(st.session_state.data[col].dtype)}" for col in columns]
                create_table_query = f"CREATE TABLE IF NOT EXISTS {name_table} ({', '.join(column_types)})"
                cursor.execute(create_table_query)
                
                # Insert data into table
                for _, row in st.session_state.data.iterrows():
                    row_data = [str(row[col]) for col in columns]
                    cursor.execute(f"INSERT INTO {name_table} ({', '.join(columns)}) VALUES ({', '.join(['?' for _ in row_data])})", row_data)
                
                # Create metadata table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        dataset_name TEXT,
                        n_rows INTEGER,
                        n_columns INTEGER,
                        size INTEGER,
                        target_column TEXT
                    )
                """)
                
                # Insert metadata
                num_rows = len(st.session_state.data)
                num_columns = len(columns)
                size = st.session_state.uploaded_file.size
                cursor.execute("INSERT INTO metadata VALUES (?, ?, ?, ?, ?)", (name_table, num_rows, num_columns, size, selected))
                
                conn.commit()
                conn.close()
                
                st.success("Data added to db")
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()