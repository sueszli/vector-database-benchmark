def test_pull(bench_dvc, tmp_dir, dvc, make_dataset, remote):
    if False:
        print('Hello World!')
    make_dataset(cache=False, dvcfile=True, files=False, remote=True)
    bench_dvc('pull')