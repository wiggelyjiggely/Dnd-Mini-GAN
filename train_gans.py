import torch
from torch.autograd.variable import Variable
import stl as stl
import torch.nn as nn
import numpy as np
import trimesh
import os
from tqdm import tqdm
from models.ddCGAN import Generator, Discriminator


class custom_dataset(torch.utils.data.Dataset):
    def __init__(self,folder):
        for foldername in os.listdir(folder):
            for filename in os.listdir(folder + "/" +foldername):
                if filename.endswith('.stl'):
                    mesh = trimesh.load(folder +"/"+ foldername + '/' + filename, force='mesh')
                    mesh = torch.tensor(mesh.vertices).float()

                    self.data = mesh
                    self.labels = filename.split('.stl')[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == '__main__':
    folder = 'thingiscrape/downloads/stls'
    dataset = custom_dataset(folder)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    noise_dim = 200 # latent space vector dim
    in_channels = 612 # convolutional channels
    dim = 64  # cube volume
    model_generator = Generator(in_channels=in_channels, out_dim=dim, out_channels=1, noise_dim=noise_dim).cuda()
    model_discriminator = Discriminator(in_channels=in_channels, dim=dim, out_conv_channels=in_channels).cuda()
    model_discriminator.train()
    model_generator.train()
    d_optimizer = torch.optim.Adam(model_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(model_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss = nn.BCELoss()
    num_epochs = 100
    
    for epoch in tqdm(range(num_epochs), total=num_epochs, leave=False, position=0):
        for n_batch, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False, position=1):
            #pad label to 100 dimensions
            label = np.pad(label, (0, 100-len(label)), 'constant')

            # train discriminator
            model_discriminator.zero_grad()
            real_data = data.cuda()
            real_label = label.cuda()
            
            # train with real data
            prediction_real = model_discriminator(real_data, real_label)
            error_real = loss(prediction_real, Variable(torch.ones(real_data.size(0))).cuda())
            error_real.backward()

            # train with fake data
            noise = torch.randn(real_data.size(0), noise_dim).cuda()
            fake_label = label.cuda()
            fake_data = model_generator(noise, fake_label)
            prediction_fake = model_discriminator(fake_data, fake_label)
            error_fake = loss(prediction_fake, Variable(torch.zeros(real_data.size(0))).cuda())
            error_fake.backward()
            d_optimizer.step()

            # train generator
            model_generator.zero_grad()
            noise = torch.randn(real_data.size(0), noise_dim).cuda()
            fake_data = model_generator(noise)
            fake_label = label.cuda()
            prediction = model_discriminator(fake_data, fake_label)
            error_generator = loss(prediction, Variable(torch.ones(real_data.size(0))).cuda())
            error_generator.backward()
            g_optimizer.step()


            # save the model
            if (n_batch) % 100 == 0:
                torch.save(model_generator.state_dict(), 'generator.pkl')
                torch.save(model_discriminator.state_dict(), 'discriminator.pkl')