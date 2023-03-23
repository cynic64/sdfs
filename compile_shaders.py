#!/usr/bin/env python3

import os

SHADER_DIR = 'shaders/'
INCLUDE_DIR = 'shaders_include/'
OUT_DIR = 'shaders_processed/'

for shader in os.listdir(SHADER_DIR):
        if not shader.endswith('.glsl'):
                print('Skipping', shader)
                print()
                continue

        print('Processing', shader)

        # Replace all #includes
        text = open(SHADER_DIR + shader, 'r').read().split('\n')

        lines_out = []
        for i, line in enumerate(text):
                if line.startswith('#include'):
                        included_path = INCLUDE_DIR + line[len('#include '):]
                        print(f'Replacing {line} on line {i} with {included_path}')
                        lines_out.append(open(included_path, 'r').read())
                else:
                        lines_out.append(line)

        processed_path = OUT_DIR + shader
        f = open(processed_path, 'w')
        f.write('\n'.join(lines_out))
        f.close()

        if '.vs.' in shader: stage = 'vertex'
        elif '.fs.' in shader: stage = 'fragment'
        else: stage = 'compute'

        command = f'glslc -fshader-stage={stage} {processed_path} -o {processed_path}.spv'
        print(command)
        os.system(command)

        print()
