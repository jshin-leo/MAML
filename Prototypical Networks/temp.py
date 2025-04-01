def inner_loop(self, support_vectors, support_labels):
    ''' BaseNet with 3-layers '''
    # temp_model = BaseNet(self.model.fc3.out_features) # create a temporary model

    # temp_model.load_state_dict(self.model.state_dict(), strict=False) # copy the weights from the main model
    # temp_model = clone_model(self.model)
    temp_model = {name: param.clone() for name, param in self.model.named_parameters()}
    t_optimizer = torch.optim.SGD(temp_model.parameters(), lr=self.inner_lr)

    for step in range(self.num_inner_steps):
        predictions = temp_model(support_vectors)
        loss = F.cross_entropy(predictions, support_labels)
        # grads = torch.autograd.grad(loss, temp_model.parameters(), create_graph=True)
        t_optimizer.zero_grad()
        loss.backward()
        t_optimizer.step()

        # [Module Test] Log the loss value --> 9/7 Checked that loss is decreasing as inner loop iterates.
        # if self.mode == 'verbose_maml':
        #   print(f"    [Inner Loop (steps {step}/{num_inner_steps})] Loss: {loss.item()}")

        # update temp_model parameters with the gradients
        '''
        with torch.no_grad():
            for param, grad in zip(temp_model.parameters(), grads):
                param -= self.inner_lr * grad
        '''

    # with torch.no_grad():
    # for param, grad in zip(temp_model.parameters(), grads):
    # param -= self.inner_lr * grad

    return temp_model