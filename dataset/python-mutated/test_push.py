def test_push(bench_dvc, tmp_dir, dvc, make_dataset, remote):
    if False:
        print('Hello World!')
    make_dataset(cache=True, dvcfile=True, files=False)
    bench_dvc('push')