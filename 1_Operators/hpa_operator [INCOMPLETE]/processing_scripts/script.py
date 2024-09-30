import pandas as pd
import argparse
import re

# Create an argument parser
parser = argparse.ArgumentParser(description='Process CSV file.')

# Add an argument for the file path
parser.add_argument('file_path', type=str, help='Path to the CSV file')

# Parse the command line arguments
args = parser.parse_args()

# Access the file path argument
file_path = args.file_path

if not file_path.endswith('.csv'):
    raise ValueError('Input file must be a CSV file.')

# Extract the number after "group" from the file name
group_number = re.search(r'group(\d+)', file_path).group(1)

# Print the extracted group number
print(f"Group number: {group_number}")


df = pd.read_csv(file_path)

# Split the 'Microservice | Replicas' column into separate 'Microservice' and 'Replicas' columns
df[['Microservice', 'Replicas']] = df['Microservice | Replicas'].str.split('|', expand=True)

# Strip any leading or trailing whitespace from the new columns
df['Microservice'] = df['Microservice'].str.strip()
df['Replicas'] = df['Replicas'].str.strip()

# Convert 'Replicas' to integer
df['Replicas'] = df['Replicas'].str.extract('(\d+)').astype(int)

# Strip any leading or trailing whitespace from column names
df.columns = df.columns.str.strip()

# Drop the original combined column
df = df.drop(columns=['Microservice | Replicas'])

# Display the cleaned dataframe
print(df.head())


