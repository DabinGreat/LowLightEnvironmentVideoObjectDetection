from .base import BaseVideoDetector
from .dark_detect import DarkDetect
# from .dark_base import DarkBase
from .dff import DFF
from .fgfa import FGFA
from .selsa import SELSA
from .llvod import LLVOD
from .slesa_dark_detect import SelsaDarkDetect
from .selsa_darkfarm_detect import SelsaDarkfarmDetect
from .selsa_clean_detect import SelsaCleanDetect
from .selsa_new_det import SelsaNewDetect
from .selsa_noise_detect import SelsaNoiseDetect
from .selsa_noise_darkfarm_detect import SelsaNoiseDarkfarmDetect
from .selsa_new_darkfarm_detect import SelsaNewDarkfarmDetect
from .selsa_clean_darkfarm_detect import SelsaCleanDarkfarmDetect
from .selsa_new_vid_detect import SelsaNewVIDDetect
from .selsa_fastdvd_darkfarm import SelsaFastDVDnetDetect


__all__ = ['BaseVideoDetector', 'DarkDetect',
           'DFF', 'FGFA', 'SELSA',
           'LLVOD', 'SelsaDarkDetect', 'SelsaDarkfarmDetect',
           'SelsaCleanDetect', 'SelsaNoiseDetect', 'SelsaNewDetect',
           'SelsaNoiseDarkfarmDetect', 'SelsaNewDarkfarmDetect',
           'SelsaCleanDarkfarmDetect', 'SelsaNewVIDDetect',
           'SelsaFastDVDnetDetect']
