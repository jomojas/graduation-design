import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, base_filters=64):
        super(Generator, self).__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 16, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 8, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        bottleneck = self.bottleneck(enc4)
        
        dec1 = self.decoder1(bottleneck)
        dec1 = torch.cat([dec1, enc4], dim=1)
        
        dec2 = self.decoder2(dec1)
        dec2 = torch.cat([dec2, enc3], dim=1)
        
        dec3 = self.decoder3(dec2)
        dec3 = torch.cat([dec3, enc2], dim=1)
        
        dec4 = self.decoder4(dec3)
        dec4 = torch.cat([dec4, enc1], dim=1)
        
        output = self.decoder5(dec4)
        
        return output


def load_model(model_path: str, device: str = "cuda") -> Generator:
    model = Generator(input_channels=1, output_channels=1)
    
    if model_path:
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"模型加载成功: {model_path}")
        except FileNotFoundError:
            print(f"警告: 模型文件未找到 {model_path}，使用随机初始化的模型")
        except Exception as e:
            print(f"警告: 模型加载失败: {e}，使用随机初始化的模型")
    
    model = model.to(device)
    model.eval()
    return model
