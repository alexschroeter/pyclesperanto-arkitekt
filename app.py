from arkitekt import register
import time
from mikro.api.schema import (
    from_xarray,
    RepresentationFragment,
)
import pyclesperanto_prototype as cle

@register
def voronoi_labeling(representation: RepresentationFragment) -> RepresentationFragment:
    """create voronoi labels

    Parameters
    ----------
    representation : RepresentationFragment

    Returns
    -------
    str
        A string with Hello {n}
    """
    image = cle.asarray(representation.data.sel(c=0, t=0).data.compute())
    binary = cle.greater_constant(image, scalar=128)
    labels = cle.voronoi_labeling(binary)
    voronoi = cle.detect_label_edges(labels)

    generated = from_xarray(
        voronoi,
        name=f"Voronoi label of {representation.name}",
        tags=["segmented"],
        origins=[representation],
    )

    return generated

