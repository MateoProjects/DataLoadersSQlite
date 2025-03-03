# SQLite Data Loader Project

A personal project I created to help me load data from SQLite databases into PyTorch for my machine learning experiments. I built this to make it easier to work with my datasets and keep track of everything in one place. The loader was designed to work with a metadata database that stores important information about each dataset, including the target column, number of rows/columns, and other metadata.

## What I Built

I made a few cool things in this project:
- A custom PyTorch Dataset class that loads data from SQLite and automatically handles target column separation based on metadata
- Automatic tracking of dataset info like size, column types, and target variables in a metadata table
- A simple web interface (using Streamlit) to upload my CSV files and specify target columns
- Basic training loop to test everything works with the metadata-aware data loading

## Why I Made This

I kept running into the same issues in my ML projects:
- Needed an easy way to load data from SQLite into PyTorch with proper target handling
- Wanted to quickly upload CSVs and start training without manual target column setup
- Got tired of manually tracking dataset metadata and target variables
- Wanted something simple that automatically handles feature/target separation based on metadata

## How I Use It

Pretty straightforward:
1. Upload my CSV through the Streamlit interface and specify the target column
2. The metadata table automatically tracks my dataset info including the target
3. Create a DataLoader that uses the metadata to properly separate features/targets
4. Train my models with correctly formatted data!

I'll probably keep adding features as I need them for my projects. Feel free to check out the code to see how the metadata-driven loading works!
