import numpy as np

def read_npz_file(file_path):
    try:
        data = np.load(file_path)
        print(f"Keys in '{file_path}': {list(data.keys())}\n")
        
        for key in data.files:
            print(f"Data under key '{key}':")
            print(data[key])
            print('-' * 40)

        data.close()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    file_path = input("Enter path to .npz file: ").strip()
    read_npz_file(file_path)

