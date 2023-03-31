import os
import subprocess
import math

# Inclusive, so that's ' ' - '~'
min_char = 32
max_char = 126

out_path = 'font.txt'

if os.path.isfile(out_path):
        os.remove(out_path)

max_height = 0
max_width = 0
chars = []
for i in range(min_char, max_char + 1):
        char = chr(i)
        result = subprocess.run(['figlet', '-f', 'banner', char], stdout=subprocess.PIPE)
        lines = result.stdout.decode('utf-8').split('\n')
        height = len(lines)
        width = max([len(x) for x in lines])
        max_height = max(height, max_height)
        max_width = max(width, max_width)

        chars.append(lines)

print(f'Max dims: {max_width} x {max_height}')

with open(out_path, 'w') as f:
        f.write(f'{max_width}\n{max_height}\n')

        for char in chars:
                for line in char:
                        # Make all lines the same length
                        chars_to_add = max_width - len(line)
                        before = math.floor(chars_to_add / 2)
                        after = math.ceil(chars_to_add / 2)
                        assert(len(line) + before + after == max_width)

                        padded = ' ' * before + line + ' ' * after
                        f.write(padded + '\n')
