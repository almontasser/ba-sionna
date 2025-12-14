"""Models module for beam alignment"""

from sionna.phy.channel.tr38901 import PanelArray
from config import Config


bs_array = PanelArray(
    num_rows_per_panel=1,
    num_cols_per_panel=Config.NTX,
    polarization="single",
    polarization_type="V",
    antenna_pattern="38.901",
    carrier_frequency=Config.CARRIER_FREQUENCY,
)

ut_array = PanelArray(
    num_rows_per_panel=1,
    num_cols_per_panel=Config.NRX,
    polarization="single",
    polarization_type="V",
    antenna_pattern="omni",
    carrier_frequency=Config.CARRIER_FREQUENCY,
)
__all__ = ["bs_array", "ut_array"]
