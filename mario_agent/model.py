import torch
import torch.nn as nn


class MarioDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_output_size = self._get_conv_output_size(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def _get_conv_output_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        output = self.conv(dummy_input)
        return int(torch.prod(torch.tensor(output.shape[1:])))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)