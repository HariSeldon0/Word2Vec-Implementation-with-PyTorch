import torch


class Trainer:

    def __init__(self, dataloader, model, epochs, device, optimizer, loss_fn):
        self.dataloader = dataloader
        self.model = model.to(device)
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self):

        for epoch in range(self.epochs):
            print(f"### epoch {epoch} ###")
            self._train_epoch()

    def _train_epoch(self):
        size = len(self.dataloader)

        self.model.train()
        for idx, (inputs, labels) in enumerate(self.dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(inputs)
            loss = self.loss_fn(pred, labels)
            loss.backward()
            self.optimizer.step()

            if idx % (size / 10) == 0:
                print(f"loss:{loss.item()}")

    def save_model(self, path):
        torch.save(self.model, path)
