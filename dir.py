import os
for file in os.listdir("./test_img"):
    file_name = os.path.splitext(file)[0]
    print(file_name)