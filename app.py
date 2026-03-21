import os

def split_zip(input_zip, num_parts, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(input_zip)
    base_name = os.path.basename(input_zip)

    file_size = os.path.getsize(input_zip)
    chunk_size = file_size // num_parts + 1

    with open(input_zip, "rb") as f:
        for i in range(1, num_parts+1):
            chunk = f.read(chunk_size)
            if not chunk:
                break
            part_filename = os.path.join(output_dir, f"{base_name}.part{i}")
            with open(part_filename, "wb") as chunk_file:
                chunk_file.write(chunk)
            print(f"Created {part_filename} ({len(chunk)} bytes)")

# --- Example: split large ZIP into 2 parts ---
file = r"C:\Users\pc lenovo\Downloads\images.zip"
split_zip(file, num_parts=2)
