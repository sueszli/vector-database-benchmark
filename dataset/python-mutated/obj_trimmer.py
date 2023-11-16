import argparse
import os
from typing import Optional, List, TextIO
'\n    Used to reduce the size of obj files used for printer platform models.\n    \n    Trims trailing 0 from coordinates\n    Removes duplicate vertex texture coordinates\n    Removes any rows that are not a face, vertex or vertex texture\n'

def process_obj(input_file: str, output_file: str) -> None:
    if False:
        while True:
            i = 10
    with open(input_file, 'r') as in_obj, open('temp', 'w') as temp:
        trim_lines(in_obj, temp)
    with open('temp', 'r') as temp, open(output_file, 'w') as out_obj:
        merge_duplicate_vt(temp, out_obj)
    os.remove('temp')

def trim_lines(in_obj: TextIO, out_obj: TextIO) -> None:
    if False:
        while True:
            i = 10
    for line in in_obj:
        line = trim_line(line)
        if line:
            out_obj.write(line + '\n')

def trim_line(line: str) -> Optional[str]:
    if False:
        return 10
    values = line.split()
    if values[0] == 'vt':
        return trim_vertex_texture(values)
    elif values[0] == 'f':
        return trim_face(values)
    elif values[0] == 'v':
        return trim_vertex(values)
    return

def trim_face(values: List[str]) -> str:
    if False:
        print('Hello World!')
    for (i, coordinates) in enumerate(values[1:]):
        (v, vt) = coordinates.split('/')[:2]
        values[i + 1] = v + '/' + vt
    return ' '.join(values)

def trim_vertex(values: List[str]) -> str:
    if False:
        for i in range(10):
            print('nop')
    for (i, coordinate) in enumerate(values[1:]):
        values[i + 1] = str(float(coordinate))
    return ' '.join(values)

def trim_vertex_texture(values: List[str]) -> str:
    if False:
        return 10
    for (i, coordinate) in enumerate(values[1:]):
        values[i + 1] = str(float(coordinate))
    return ' '.join(values)

def merge_duplicate_vt(in_obj, out_obj):
    if False:
        while True:
            i = 10
    vt_index_mapping = {}
    vt_to_index = {}
    vt_index = 1
    skipped_count = 0
    for line in in_obj.readlines():
        if line[0] == 'f':
            continue
        if line[:2] == 'vt':
            if line in vt_to_index.keys():
                vt_index_mapping[vt_index] = vt_to_index[line]
                skipped_count += 1
            else:
                vt_to_index[line] = vt_index - skipped_count
                vt_index_mapping[vt_index] = vt_index - skipped_count
                out_obj.write(line)
            vt_index += 1
        else:
            out_obj.write(line)
    in_obj.seek(0)
    for line in in_obj.readlines():
        if line[0] != 'f':
            continue
        values = line.split()
        for (i, coordinates) in enumerate(values[1:]):
            (v, vt) = coordinates.split('/')[:2]
            vt = int(vt)
            if vt in vt_index_mapping.keys():
                vt = vt_index_mapping[vt]
            values[i + 1] = v + '/' + str(vt)
        out_obj.write(' '.join(values) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reduce the size of a .obj file')
    parser.add_argument('input_file', type=str, help='Input .obj file name')
    parser.add_argument('--output_file', default='output.obj', type=str, help='Output .obj file name')
    args = parser.parse_args()
    process_obj(args.input_file, args.output_file)