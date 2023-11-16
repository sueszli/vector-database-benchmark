def basic_iteration_step(self, ddp_model, criterion, optimizer, hook_state, epoch, index, batch):
    if False:
        print('Hello World!')
    '\n    A function that performs an iteration of training.\n    Args:\n        ddp_model (nn.Module): distributed data parallel model\n        criterion (nn.Module): loss function to measure model\n        optimizer (optim.Optimizer): updates model parameters\n        hook_state (object): ddp communication hook state object\n        epoch (int): index of pass through the data\n        index (int): iteration number - 1 in current batch\n        batch (list): training examples\n    '
    hook_state.next_batch()
    self.record_batch_start(self.epoch_key(epoch, index))
    optimizer.zero_grad()
    self.record_forward_start(self.epoch_key(epoch, index))
    loss = criterion(ddp_model(batch[0]), batch[1])
    self.record_forward_end(self.epoch_key(epoch, index))
    self.record_backward_start(self.epoch_key(epoch, index))
    loss.backward()
    self.record_backward_end(self.epoch_key(epoch, index))
    optimizer.step()
    self.record_batch_end(self.epoch_key(epoch, index))