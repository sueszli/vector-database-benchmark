def db_input(model, blobs_out, batch_size, db, db_type):
    if False:
        print('Hello World!')
    dbreader_name = 'dbreader_' + db
    dbreader = model.param_init_net.CreateDB([], dbreader_name, db=db, db_type=db_type)
    return model.net.TensorProtosDBInput(dbreader, blobs_out, batch_size=batch_size)