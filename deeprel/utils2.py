import logging
import os

import GPUtil


def pick_device():
    try:
        GPUtil.showUtilization()
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list
        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
        print('Device ID (unmasked): ' + str(DEVICE_ID))
    except:
        logging.exception('Cannot detect GPUs')