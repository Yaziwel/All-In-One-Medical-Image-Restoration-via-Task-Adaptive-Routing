import os

def capitalize_first_letter(folder_path):
    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 检查路径是否是文件而不是子文件夹
        if os.path.isfile(file_path):
            # 获取文件名的首字母并将其转换为大写
            new_filename = filename[0].upper() + filename[1:]

            # 构建新的文件路径
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"已将文件 '{filename}' 重命名为 '{new_filename}'")

for dose in ["10s", "20s", "30s", "40s", "60s", "120s"]:
    # 指定要处理的文件夹路径
    folder_path = "/home/data/zhiwen/dataset/MC-NC/DMI/NIFTI/{}".format(dose)
    
    # 调用函数进行文件名修改
    capitalize_first_letter(folder_path)
