import torch

def tv_loss(input_t):
	temp1 = torch.cat((input_t[:, :, 1:, :], input_t[:, :, -1, :].unsqueeze(2)), 2)
	temp2 = torch.cat((input_t[:, :, :, 1:], input_t[:, :, :, -1].unsqueeze(3)), 3)
	temp = (input_t - temp1)**2 + (input_t - temp2)**2
	return temp.sum()

if __name__ == '__main__':
	input = torch.rand(4,3,32,32)
	print(tv_loss(input))
