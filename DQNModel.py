#!/usr/bin/env python3
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.InstanceNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        
        self.bn2 = nn.InstanceNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.InstanceNorm1d(512)
        self.fc4 = nn.Linear(512, 1024)
        self.bn4 = nn.InstanceNorm1d(1024)
        self.fc_x = nn.Linear(1024, 1)
        self.fc_y = nn.Linear(1024, 1)
        self.fc_z = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))

        x_speed_x = torch.tanh(self.fc_x(x))
        x_speed_y = torch.tanh(self.fc_y(x))
        x_speed_z = torch.tanh(self.fc_z(x))

        return torch.cat((x_speed_x, x_speed_y, x_speed_z), dim=1)

# class DQN(nn.Module):
#     def __init__(self, input_channels=1):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(192, 512)
#         self.fc2 = nn.Linear(512, 3)  # 3 çıkış, her biri için x, y ve z hızlarını temsil eder
#         self.fc2_x = nn.Linear(512, 1)  # x hızı için çıkış
#         self.fc2_y = nn.Linear(512, 1)  # y hızı için çıkış
#         self.fc2_z = nn.Linear(512, 1)  # z hızı için çıkış
        
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = torch.relu(self.conv3(x))
#         x = x.view(x.size(0), -1)  # Flatten işlemi
#         x = torch.relu(self.fc1(x))
#         x_speed_x = self.fc2_x(x)  # x hızı için çıkış
#         x_speed_y = self.fc2_y(x)  # y hızı için çıkış
#         x_speed_z = self.fc2_z(x)  # z hızı için çıkış
#         # x = self.fc2(x)
#         # x = x.unsqueeze(1)
#         return torch.cat((x_speed_x, x_speed_y, x_speed_z), dim=1)#.unsqueeze(1)


# model = DQN()

# # Örnek bir girdi tensoru oluştur (batch_size=1, kanal_sayısı=1, uzunluk=3)
# input_vector = torch.tensor([[0.0, 0.0, 10.0], [0.4, 0.5, 0.6], [0.0, 0.0, 10.0]], dtype=torch.float32)

# # input_vector = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# print("girdiler:", input_vector)
# print("Girdilerin boyutu:", input_vector.size())
# # Modeli kullanarak çıktıları elde et
# output = model(input_vector)

# # Çıktıları ve boyutunu yazdır
# print("Çıktılar:", output)
# print("Çıktıların boyutu:", output.size())

# with torch.no_grad():
#     output = model(input_vector)

# # Tahminlerin boyutunu kontrol et
# # print("Tahminlerin boyutu:", output.size())

# # # Tahminleri yazdır
# # print("Tahminler:", output)