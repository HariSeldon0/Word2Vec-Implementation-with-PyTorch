import torch


class Trainer:

    def __init__(
        self,
        dataloader,
        model,
        epochs,
        device,
        optimizer,
        loss_fn,
        log_file="loss_log.txt",
    ):
        self.dataloader = dataloader
        self.model = model.to(device)
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.log_file = log_file

        # Clear the log file at the beginning
        with open(self.log_file, "w") as f:
            f.write("Epoch,Iteration,Loss\n")

    def train(self):
        for epoch in range(self.epochs):
            print(f"### epoch {epoch} ###")
            self._train_epoch(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        for idx, (inputs, labels) in enumerate(self.dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(inputs)
            loss = self.loss_fn(pred, labels)
            loss.backward()
            self.optimizer.step()

            loss_value = loss.item()
            print(f"loss:{loss_value}")

            # Save loss to file
            with open(self.log_file, "a") as f:
                f.write(f"{epoch},{idx},{loss_value}\n")

    def save_model(self, path):
        torch.save(self.model, path)
