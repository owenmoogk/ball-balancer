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
    C_MATRIX = np.array([3063.894803087576,0.0,448.9224830647149],
                        [0.0,2604.697840954291,988.3923029245163],
                        [0.0, 0.0, 1.0])
    
    D_PARAM = np.array([-2.348290496607099,
                        4.3826225970761605,
                        -0.18277748843460256,
                        0.030430158578043043,
                        -7.934750868203259])