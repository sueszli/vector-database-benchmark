def cp(src_path, dst_path):
    if False:
        for i in range(10):
            print('nop')
    with open(src_path) as src, open(dst_path, mode='w') as dst:
        contents = src.read()
        dst.write(contents)