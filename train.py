import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import utils
import os

from model import Generator
from model import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/chinese/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=80, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--lamda', type=int, default=10,help='cycle regularization parameter')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--DA', type=str, default='output/netD_A.pth', help='Dis_A checkpoint file')
parser.add_argument('--DB', type=str, default='output/netD_B.pth', help='D_isB  checkpoint file')
parser.add_argument('--resume', type=bool, default=False, help='resume training from checkpoint')
opt = parser.parse_args()

netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    
    
if opt.resume:
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
    netD_A.load_state_dict(torch.load(opt.DA))
    netD_B.load_state_dict(torch.load(opt.DB))
else:
    netG_A2B.apply(utils.weights_init_normal)
    netG_B2A.apply(utils.weights_init_normal)
    netD_A.apply(utils.weights_init_normal)
    netD_B.apply(utils.weights_init_normal)

MSE = torch.nn.MSELoss()
L1= torch.nn.L1Loss()

#Optimizer
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))

optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G= torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=utils.LambdaLR(opt.n_epochs,opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A= torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=utils.LambdaLR(opt.n_epochs,opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B= torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=utils.LambdaLR(opt.n_epochs,opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, 256, 256)
input_B = Tensor(opt.batchSize, opt.output_nc, 256, 256)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = utils.ReplayBuffer()
fake_B_buffer = utils.ReplayBuffer()

#### load data
transform = [transforms.Resize(int(256*1.12),Image.BICUBIC),
             transforms.RandomCrop(256),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    
dataloader = DataLoader(utils.ImageDataset(opt.dataroot, transforms_=transform, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)



#### Training
for epoch in range(opt.epoch, opt.n_epochs):
    for i,batch in enumerate(dataloader):
        print(epoch,i,len(dataloader))
        
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        
        optimizer_G.zero_grad()


        # Forward pass through generators
##################################################
        fake_B = netG_A2B(real_A)
        fake_A = netG_B2A(real_B)
        recovered_A = netG_B2A(fake_B)
        recovered_B = netG_A2B(fake_A)
        ### Identity: A to B should be equal to b is real b is fed
        id_B = netG_A2B(real_B)
        id_A = netG_B2A(real_A)
        
        ## GAN loss
        loss_GAN_A2B = MSE(netD_B(fake_B),target_real)
        loss_GAN_B2A = MSE(netD_A(fake_A),target_real)
        ## Cycle Loss
        loss_cycle_ABA = L1(recovered_A,real_A)*opt.lamda
        loss_cycle_BAB = L1(recovered_B,real_B)*opt.lamda
        ## Identity Loss
        loss_identity_B = L1(id_B, real_B)*opt.lamda*0.5
        loss_identity_A = L1(id_A, real_A)*opt.lamda*0.5
        
        ## Total Loss for generator:
        loss_G = loss_GAN_A2B+loss_GAN_B2A+loss_cycle_ABA+loss_cycle_BAB +loss_identity_B+loss_identity_A
        loss_G.backward()
        optimizer_G.step()
        
       
        
        # Discriminator B
        optimizer_D_B.zero_grad()
        pred_real = netD_B(real_B)  #real loss
        loss_D_real = MSE(pred_real,target_real)
        
        # Fake loss  
        fake_B = fake_B_buffer.push_and_pop(fake_B) #sampling
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = MSE(pred_fake, target_fake)
        
        #total
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()
        optimizer_D_B.step()
        
##################################################    
        
        # Discriminator A
        optimizer_D_A.zero_grad()
        pred_real = netD_A(real_A)  #real loss
        loss_D_real = MSE(pred_real,target_real)
        
        # Fake loss  
        fake_A = fake_A_buffer.push_and_pop(fake_A) #sampling
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = MSE(pred_fake, target_fake)
        
        #total
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()
        optimizer_D_A.step()
        
        print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
                                            (epoch, i + 1,len(dataloader),loss_G,loss_D_B+loss_D_B))
        

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    # Save models checkpoints
    if not os.path.exists('output'):
        os.makedirs('output')
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')
    if epoch % 5 == 0:
      torch.save(netG_A2B.state_dict(), 'output/netG_A2B'+str(epoch)+'.pth')
      torch.save(netG_B2A.state_dict(), 'output/netG_B2A'+str(epoch)+'.pth')
      torch.save(netD_A.state_dict(), 'output/netD_A'+str(epoch)+'.pth')
      torch.save(netD_B.state_dict(), 'output/netD_B'+str(epoch)+'.pth')
      print('saved test checkpoints')

###################################


#