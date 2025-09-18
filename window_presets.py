# Presets for different CT tissue types
WINDOW_PRESETS = {
    "soft_tissue": {"center": 40, "width": 400},
    "lung": {"center": -600, "width": 1500},
    "bone": {"center": 500, "width": 2000},
    "brain": {"center": 40, "width": 80},
    "liver": {"center": 60, "width": 150},
    "abdomen": {"center": 60, "width": 400},
    # Global HU-Clip preset: [-1000, 2000] ⇒ center=500, width=3000
    "global": {"center": 500, "width": 3000},
    "default": {"center": 40, "width": 400},  # fallback
}
