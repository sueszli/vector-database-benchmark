import polars as pl
df = pl.DataFrame({'keys': ['a', 'a', 'b'], 'values': [10, 7, 1]})
out = df.group_by('keys', maintain_order=True).agg(pl.col('values').map_batches(lambda s: s.shift()).alias('shift_map'), pl.col('values').shift().alias('shift_expression'))
print(df)
out = df.group_by('keys', maintain_order=True).agg(pl.col('values').map_elements(lambda s: s.shift()).alias('shift_map'), pl.col('values').shift().alias('shift_expression'))
print(out)
counter = 0

def add_counter(val: int) -> int:
    if False:
        return 10
    global counter
    counter += 1
    return counter + val
out = df.select(pl.col('values').map_elements(add_counter).alias('solution_apply'), (pl.col('values') + pl.int_range(1, pl.count() + 1)).alias('solution_expr'))
print(out)
out = df.select(pl.struct(['keys', 'values']).map_elements(lambda x: len(x['keys']) + x['values']).alias('solution_apply'), (pl.col('keys').str.len_bytes() + pl.col('values')).alias('solution_expr'))
print(out)