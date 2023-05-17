__version__ = "0.0.1"
from ._widget import ExampleQWidget, example_magic_widget
import napari


__all__ = (
    "ExampleQWidget",
    "example_magic_widget",
)

viewer = napari.current_viewer()