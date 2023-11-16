from panda3d.core import GeomVertexArrayFormat, GeomVertexFormat, Geom

def test_format_arrays():
    if False:
        print('Hello World!')
    array1 = GeomVertexArrayFormat('vertex', 3, Geom.NT_float32, Geom.C_point)
    array2 = GeomVertexArrayFormat('normal', 3, Geom.NT_float32, Geom.C_normal)
    array3 = GeomVertexArrayFormat('color', 4, Geom.NT_float32, Geom.C_color)
    array4 = GeomVertexArrayFormat('texcoord', 2, Geom.NT_float32, Geom.C_texcoord)
    assert array1.get_ref_count() == 1
    assert array2.get_ref_count() == 1
    assert array3.get_ref_count() == 1
    assert array4.get_ref_count() == 1
    format = GeomVertexFormat()

    def expect_arrays(*args):
        if False:
            return 10
        assert format.get_num_arrays() == len(args)
        assert len(format.arrays) == len(args)
        assert tuple(format.arrays) == args
        arrays = format.get_arrays()
        assert tuple(arrays) == args
        assert array1.get_ref_count() == 1 + arrays.count(array1) * 2
        assert array2.get_ref_count() == 1 + arrays.count(array2) * 2
        assert array3.get_ref_count() == 1 + arrays.count(array3) * 2
        assert array4.get_ref_count() == 1 + arrays.count(array4) * 2
    expect_arrays()
    format.add_array(array1)
    expect_arrays(array1)
    format.add_array(array2)
    expect_arrays(array1, array2)
    format.add_array(array3)
    expect_arrays(array1, array2, array3)
    format.add_array(array4)
    expect_arrays(array1, array2, array3, array4)
    assert format.get_num_arrays() == 4
    assert len(format.arrays) == 4
    assert tuple(format.get_arrays()) == (array1, array2, array3, array4)
    format.remove_array(0)
    expect_arrays(array2, array3, array4)
    format.remove_array(0)
    expect_arrays(array3, array4)
    format.remove_array(0)
    expect_arrays(array4)
    format.remove_array(0)
    expect_arrays()
    format.insert_array(0, array1)
    expect_arrays(array1)
    format.insert_array(1, array2)
    expect_arrays(array1, array2)
    format.insert_array(2, array3)
    expect_arrays(array1, array2, array3)
    format.insert_array(3, array4)
    expect_arrays(array1, array2, array3, array4)
    format.remove_array(3)
    expect_arrays(array1, array2, array3)
    format.remove_array(2)
    expect_arrays(array1, array2)
    format.remove_array(1)
    expect_arrays(array1)
    format.remove_array(0)
    expect_arrays()
    format.insert_array(0, array4)
    expect_arrays(array4)
    format.insert_array(0, array3)
    expect_arrays(array3, array4)
    format.insert_array(0, array2)
    expect_arrays(array2, array3, array4)
    format.insert_array(0, array1)
    expect_arrays(array1, array2, array3, array4)
    format.remove_array(2)
    expect_arrays(array1, array2, array4)
    format.insert_array(2, array3)
    expect_arrays(array1, array2, array3, array4)
    format.remove_array(1)
    expect_arrays(array1, array3, array4)
    format.remove_array(1)
    expect_arrays(array1, array4)
    format.insert_array(1, array2)
    expect_arrays(array1, array2, array4)
    format.insert_array(2, array3)
    expect_arrays(array1, array2, array3, array4)
    format.clear_arrays()
    expect_arrays()
    format.insert_array(2147483647, array1)
    expect_arrays(array1)
    format.insert_array(2147483647, array2)
    expect_arrays(array1, array2)
    format.insert_array(2147483647, array3)
    expect_arrays(array1, array2, array3)
    format.insert_array(2147483647, array4)
    expect_arrays(array1, array2, array3, array4)