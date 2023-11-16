"""Configuration parameters for RaggedTensors."""

def auto_cast_partition_dtype():
    if False:
        while True:
            i = 10
    'Whether incompatible row-partitioning dtypes should be auto-converted.\n\n  If true, then operations that combine RaggedTensors but have different\n  row-partitioning tensor dtypes will be automatically cast to a\n  compatible dtype (`tf.int64`).  If false, then such operations will result\n  in an error.\n\n  Returns:\n    `bool`\n  '
    return False