import numpy as np

__BASE_RADIUS__ = 0.05
__PLATFORM_RADIUS__ = 0.15
class Settings:
    MOTOR_LINK_LEN = 0.08
    PUSH_LINK_LEN = 0.095
    BALL_RADIUS = 0.02
    G = 9.81
    TABLE_HEIGHT = 0.10
    PLATFORM_RADIUS = __PLATFORM_RADIUS__
    BASE_RADIUS = __BASE_RADIUS__
    PLATFORM_THICKNESS = 0.003
    angles = np.deg2rad([0, 120, 240])
    BASES = [(__BASE_RADIUS__ * np.cos(a), __BASE_RADIUS__ * np.sin(a), 0.0) for a in angles]
    CONTACTS = [
            (__PLATFORM_RADIUS__ * np.cos(a), __PLATFORM_RADIUS__ * np.sin(a)) for a in angles
    ]

    # SIMULATION SETTINGS
    DT: float = 0.005
    TARGET_FPS: int = 60
    I_SPHERE: float = (2.0 / 5.0) * BALL_RADIUS**2

    # CAMERA SETTINGS
    # For Logitech C920 HD Pro Webcam
    RESO = 1
    if RESO == 1:
        # (1280 * 720)
        C_MATRIX = np.array([[932.0900126242127,0,638.6549019259521],
                                [0,931.4949715786727,342.9911521888232],
                                [0,0,1]])
        
        D_COEFFS = np.array([0.10393692155716833,
                                -0.17168576648599584,
                                -0.0005549638588882746,
                                -0.0009705263907376315,
                                0.03137706338566113])
    else:
        # (1920 * 1080)
        C_MATRIX = np.array([[1398.135018936319,0,957.9823528889282],
                                [0,1397.2424573680091,514.4867282832348],
                                [0,0,1]])
        
        D_COEFFS = np.array([0.10393692155716833,
                                -0.17168576648599584,
                                -0.0005549638588882746,
                                -0.0009705263907376315,
                                0.03137706338566113])