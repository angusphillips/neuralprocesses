from plum import convert

from .lik import construct_likelihood
from ..util import register_model

__all__ = ["construct_convgnp"]


@register_model
def construct_convgnp(nps):
    def construct_convgnp(
        dim_x=1,
        dim_y=1,
        dim_yc=None,
        dim_yt=None,
        points_per_unit=64,
        margin=0.1,
        likelihood="lowrank",
        unet_channels=(64,) * 6,
        num_basis_functions=512,
        scale=None,
        dtype=None,
    ):
        dim_yc = convert(dim_yc or dim_y, tuple)
        dim_yt = dim_yt or dim_y
        # `len(dim_yc)` is equal to the number of density channels.
        unet_in_channels = sum(dim_yc) + len(dim_yc)
        unet_out_channels, likelihood = construct_likelihood(
            nps,
            spec=likelihood,
            dim_y=dim_yt,
            num_basis_functions=num_basis_functions,
            dtype=dtype,
        )
        unet = nps.UNet(
            dim=dim_x,
            in_channels=unet_in_channels,
            out_channels=unet_out_channels,
            channels=unet_channels,
            dtype=dtype,
        )
        disc = nps.Discretisation(
            points_per_unit=points_per_unit,
            multiple=2**unet.num_halving_layers,
            margin=margin,
            dim=dim_x,
        )
        if scale is None:
            scale = 2 / disc.points_per_unit
        return nps.Model(
            nps.FunctionalCoder(
                disc,
                nps.Chain(
                    nps.PrependDensityChannel(),
                    nps.SetConv(scale, dtype=dtype),
                    nps.DivideByFirstChannel(),
                ),
            ),
            nps.Chain(
                unet,
                nps.SetConv(scale, dtype=dtype),
                likelihood,
            ),
        )

    return construct_convgnp
