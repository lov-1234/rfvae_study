import torch
import torch.nn as nn
from einops import rearrange

# Encoder Decoder Architecture

class Encoder(nn.Module):
    def __init__(
        self,
        input_channels,
        input_size,
        latent_dims,
        hidden_channels,
        num_downsampling_layers,
        num_fc_layers,
        out_fc_features,
        kernel_size=4,
        stride=2,
        padding=1
    ):
        super().__init__()

        assert num_downsampling_layers > 0, "num_downsampling_layers must be greater than 0"
        assert num_fc_layers > 0, "num_fc_layers must be greater than 0"

        self.input_channels = input_channels
        self.latent_dims = latent_dims
        self.hidden_channels = hidden_channels
        self.num_downsampling_layers = num_downsampling_layers
        self.num_fc_layers = num_fc_layers

        conv_layers = []
        for i in range(num_downsampling_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            conv_layers.append(
                nn.Conv2d(in_ch, hidden_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding)
            )
            # conv_layers.append(nn.BatchNorm2d(hidden_channels))
            conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, *input_size)
            dummy_conv_output = self.conv_layers(dummy_input)
            self.conv_out_dim = dummy_conv_output.shape[1:]
            self.output_shape = self.flatten(dummy_conv_output).shape[1]

        fc_layers = []
        in_features = self.output_shape
        for _ in range(num_fc_layers):
            fc_layers.append(nn.Linear(in_features, out_fc_features))
            # fc_layers.append(nn.BatchNorm1d(out_fc_features))
            fc_layers.append(nn.ReLU())
            in_features = out_fc_features
        self.fc_layers = nn.Sequential(*fc_layers)

        self.mu = nn.Linear(in_features, latent_dims)
        self.logvar = nn.Linear(in_features, latent_dims)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        latent_dim,
        hidden_channels,
        num_upsampling_layers,
        out_conv_shape,
        num_fc_layers,
        fc_layers_dim,
        conv_output_shape,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=0
    ):
        super().__init__()

        assert num_fc_layers > 0, "num_fc_layers must be greater than 0"
        assert num_upsampling_layers > 0, "num_upsampling_layers must be greater than 0"
        assert out_conv_shape[0] == hidden_channels, "out_conv_shape[0] must equal hidden_channels"

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.num_upsampling_layers = num_upsampling_layers
        self.input_channel = input_channel
        self.out_conv_shape = out_conv_shape

        fc_layers = []
        for i in range(num_fc_layers):
            if i == 0:
                fc_layers.append(nn.Linear(latent_dim, fc_layers_dim))
            else:
                fc_layers.append(nn.Linear(fc_layers_dim, fc_layers_dim))
            # fc_layers.append(nn.BatchNorm1d(fc_layers_dim))
            fc_layers.append(nn.ReLU())

        fc_layers.append(nn.Linear(fc_layers_dim, conv_output_shape))
        self.fc_layers = nn.Sequential(*fc_layers)

        conv_transpose_layers = []
        for u in range(num_upsampling_layers):
            is_last = (u == num_upsampling_layers - 1)

            if is_last:
                conv_transpose_layers.append(
                    nn.ConvTranspose2d(
                        hidden_channels,
                        input_channel,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding
                    )
                )
            else:
                conv_transpose_layers.append(
                    nn.ConvTranspose2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding
                    )
                )
                # conv_transpose_layers.append(nn.BatchNorm2d(hidden_channels))
                conv_transpose_layers.append(nn.ReLU())

        self.conv_transpose_layers = nn.Sequential(*conv_transpose_layers)

    def forward(self, z):
        x = self.fc_layers(z)
        x = rearrange(
            x,
            'b (c h w) -> b c h w',
            c=self.out_conv_shape[0],
            h=self.out_conv_shape[1],
            w=self.out_conv_shape[2]
        )
        x = self.conv_transpose_layers(x)
        return x

# Building the VAE


class VAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=10, hidden_channels=None,
                 num_fc_layers=None, num_upsampling_layers=None, num_downsampling_layers=None, out_fc_features=None, input_size=(64, 64), kernel_size=3, output_padding=1):
        super().__init__()
        self.encoder = Encoder(input_channels=input_channels,
                               latent_dims=latent_dim,
                               hidden_channels=hidden_channels,
                               num_fc_layers=num_fc_layers,
                               out_fc_features=out_fc_features,
                               input_size=input_size,
                               kernel_size=kernel_size,
                               num_downsampling_layers=num_downsampling_layers)
        self.decoder = Decoder(
            input_channel=input_channels,
            latent_dim=latent_dim,
            out_conv_shape=self.encoder.conv_out_dim,
            conv_output_shape=self.encoder.output_shape,
            num_fc_layers=num_fc_layers,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            output_padding=output_padding,
            num_upsampling_layers=num_upsampling_layers,
            fc_layers_dim=out_fc_features
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z
