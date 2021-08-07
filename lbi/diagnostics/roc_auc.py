import torch
import random




class ROC:
    def __init__(self, data, context, widths=[512, 512], num_classifiers=4,
                 max_steps=int(1e5), batch_size=64, validation_split=0.15, num_workers=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers

        self.classifiers = [self.make_classifier(self, widths) for _ in range(num_classifiers)]
        self.optimizers = [torch.optim.AdamW(classifier.parameters()) for classifier in self.classifiers]

        training_loader, validation_loader = self.prep_data(data, context)
        self.training_loader = training_loader
        self.validation_loader = validation_loader

    def prep_data(self, data, context):
        mask_split = torch.rand(data.shape[0]) > self.validation_split
        training_data = self.data[mask_split]
        training_context = self.context[mask_split]
        validation_data = self.data[~mask_split]
        validation_context = self.context[~mask_split]

        mu_data = torch.mean(training_data)
        mu_context = torch.mean(training_context)
        sigma_data = torch.std(training_data)
        sigma_context = torch.std(training_context)

        training_data = (training_data - mu_data)/sigma_data
        training_context = (training_context - mu_context)/sigma_context
        validation_data = (validation_data - mu_data)/sigma_data
        validation_context = (validation_context - mu_context)/sigma_context


        train_dset = torch.utils.data.TensorDataset( training_data.float(), training_context.float())
        training_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers)
        valid_dset = torch.utils.data.TensorDataset(validation_data.float(), validation_context.float())
        validation_loader = torch.utils.data.DataLoader(
            valid_dset,
            batch_size=10*self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers)
        pass

    def get_batch(self, shuffle=True):
        data_A, context_A = next(self.training_loader)
        data_B, context_B = next(self.training_loader)

        ones = torch.ones(data_A.shape[0])
        zeros = torch.zeros(data_A.shape[0])

        A = torch.cat([data_A, context_A], dim=1)
        B = torch.cat([data_B, context_B], dim=1)
        C = torch.cat([data_A, context_B], dim=1)
        D = torch.cat([data_B, context_A], dim=1)

        batch = torch.cat([A, B, C, D], dim=0)
        labels = torch.cat([ones, ones, zeros, zeros], dim=0)

        return batch, labels


    def train(self, model, optimizer):
        for i in range(self.max_steps):
            optimizer.zero_grad()
            batch, labels = self.get_batch()
            preds = model.forward(batch)
            l = self.loss(preds, labels)
            l.backward()
            optimizer.step()



        pass

    def make_classifier(self, widths):


    def run(self):
        for classifier in self.classifiers:
            self.train(classifier)