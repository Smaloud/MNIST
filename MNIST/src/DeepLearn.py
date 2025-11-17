import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch


# # 1. 检查CUDA是否可用
# print("CUDA可用:", torch.cuda.is_available())  # 应返回True

# # 2. 查看显卡数量和名称
# print("显卡数量:", torch.cuda.device_count())  # 至少为1
# print("当前显卡:", torch.cuda.current_device())  # 通常为0
# print("显卡名称:", torch.cuda.get_device_name(0))  # 显示具体型号

# # 3. 验证CUDA计算（创建张量并在GPU上运算）
# if torch.cuda.is_available():
#     # 在GPU上创建一个随机张量
#     x = torch.rand(3, 3).cuda()
#     # 执行简单计算（矩阵乘法）
#     y = x @ x.T  # 等价于x.matmul(x.T)
#     print("GPU计算结果:\n", y)
#准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.MNIST(root="../data",train = True, transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.MNIST(root="../data",train = False, transform=torchvision.transforms.ToTensor(),download=True)

#数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集长度：{}".format(train_data_size))
print("测试数据集长度：{}".format(test_data_size))



#加载数据集
train_dataloader = DataLoader(train_data, batch_size=8192)
test_dataloader = DataLoader(test_data, batch_size=8192)
#创建网络
class LeNet(nn.Module):
    def __init__(self) :
        super(LeNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,6,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5,120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84,10)

    #前向传播函数
    def forward(self,x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 16 * 5 * 5)
            x = self.fc1(x)
            x = self.fc2(x)
            output = self.fc3(x)

            return output
myLeNet = LeNet()
if torch.cuda.is_available():
    myLeNet = myLeNet.cuda()

if torch.cuda.is_available():	# cuda可用时可置于GPU中训练网络，以加快计算速度
    myLeNet.cuda()
    print("cuda is available, and the calculation will be moved to GPU\n")
else:
    print("cuda is unavailable!")

#训练和测试
optimizer = torch.optim.RMSprop(myLeNet.parameters(),lr = 1e-3)
loss_func = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_func = loss_func.cuda()

#设置训练的一些参数
total_train_step = 0
total_test_step = 0
opoch = 10
#添加Tensorboard
writer = SummaryWriter("../logs_train")
for i in range(opoch):
    print("------第{}轮训练开始------".format(i+1))    
#训练开始
    myLeNet.train()
    for batch_idx, data in enumerate(train_dataloader):
        imgs, target = data
        # 只在第一个批次展示图片（避免日志过大）
        if batch_idx == 0 and i == 0:  # i 是当前轮次（for i in range(epoch)中的i）
            # 从批次中取前 8 张图片展示
            for img_idx in range(8):
                # 图片标签："train_images/第X轮_第Y张"
                tag = f"train_images/epoch_{i+1}_img_{img_idx}"
                # 添加单张图片：add_image(标签, 图片张量, 全局步数)
                # 注意：图片张量需为 (C, H, W) 格式，MNIST 满足此格式
                writer.add_image(tag, imgs[img_idx], global_step=total_train_step)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            target = target.cuda()
        outputs = myLeNet(imgs)
        loss = loss_func(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    #测试步骤
    myLeNet.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,target = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                target = target.cuda()
            outputs =  myLeNet(imgs)
            loss = loss_func(outputs, target)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == target).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试Loss:{}".format(total_test_loss))
    print("整体测试准确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1
    torch.save(myLeNet,"myLeNet.pth".format(i))
    print("模型已保存")

writer.close()
