import os

# Tentukan path folder
folder_path = 'dataset/'
# Dapatkan daftar semua file dalam folder
file_list = os.listdir(folder_path)

# Urutkan daftar file
sorted_files = sorted(file_list)

# Tentukan nomor awal urutan
nomor_urutan = 1

# Loop melalui setiap file dalam folder
for file_name in sorted_files:
    # Dapatkan ekstensi file (jika ada)
    file_name_parts = os.path.splitext(file_name)
    extension = file_name_parts[1]

    # Bentuk nama file baru dengan format "file{nomor_urutan}{ekstensi}"
    new_name = f"nanas_matang{nomor_urutan}{extension}"
    
    # Dapatkan path lengkap dari file lama dan file baru
    old_path = os.path.join(folder_path, file_name)
    new_path = os.path.join(folder_path, new_name)
    
    # Ubah nama file
    os.rename(old_path, new_path)
    print(f"File '{file_name}' telah diubah menjadi '{new_name}'")

    # Tingkatkan nomor urutan untuk file berikutnya
    nomor_urutan += 1