import neuralprocesses as nps  # This fixes inspection below.
from .lik import construct_likelihood
from ..util import register_model

__all__ = ["construct_agnp"]


@register_model
def construct_agnp(
    dim_x=1,
    dim_y=1,
    dim_embedding=256,
    num_heads=16,
    num_enc_layers=6,
    num_dec_layers=6,
    likelihood="lowrank",
    num_basis_functions=512,
    dtype=None,
    nps=nps,
):
    """An Attentive Gaussian Neural Process.

    Args:
        dim_x (int, optional): Dimensionality of the inputs. Defaults to `1`.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to `1`.
        dim_embedding (int, optional): Dimensionality of the embedding. Defaults to
            `256`.
        num_heads (int, optional): Number of heads. Defaults to `16`.
        num_enc_layers (int, optional): Number of layers in the encoder. Defaults to
            `6`.
        num_dec_layers (int, optional): Number of layers in the decoder. Defaults to
            `6`.
        likelihood (str, optional): Likelihood. Must be one of "het", "lowrank", or
            "lowrank-correlated". Defaults to "lowrank".
        num_basis_functions (int, optional): Number of basis functions for the
            low-rank likelihood. Defaults to `512`.
        dtype (dtype, optional): Data type.

    Returns:
        :class:`.model.Model`: GNP model.
    """
    mlp_out_channels, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=dim_y,
        num_basis_functions=num_basis_functions,
        dtype=dtype,
    )
    encoder = nps.Chain(
        nps.Parallel(
            nps.InputsCoder(),
            nps.Attention(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_embedding=dim_embedding,
                num_heads=num_heads,
                num_enc_layers=num_enc_layers,
                dtype=dtype,
            ),
        ),
    )
    decoder = nps.Chain(
        nps.Materialise(),
        nps.MLP(
            in_dim=dim_x + dim_embedding,
            layers=(512,) * num_dec_layers,
            out_dim=mlp_out_channels,
            dtype=dtype,
        ),
        likelihood,
    )
    return nps.Model(encoder, decoder)