import os
import re
from collections import defaultdict

files = os.listdir()
pattern = re.compile(r'_(\d+\.\d+)_')
grouped_data = defaultdict(list)


for filename in files:
    if filename.endswith('.txt'):
        match = pattern.search(filename)
        if match:
            key = match.group(1)  # This is the learning rate or mean_eta value
            breakpoint()
            with open(filename, 'r') as file:
                grouped_data[key].append(file.read())



# # Write grouped data to new files
# for key, contents in grouped_data.items():
#     with open(f'{key}_combined.txt', 'w') as output_file:
#         output_file.write('\n'.join(contents))

print("Files have been successfully grouped and written to new files.")
