import logging
import os
from datetime import datetime 
import sys

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" 
LOG_PATH=os.path.join(os.getcwd(),"logs", LOG_FILE)
os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH=os.path.join(LOG_PATH, LOG_FILE)

logging.basicConfig(filename=LOG_FILE_PATH, format='%(asctime)s %(levelname)s %(name)s %(message)s', 
                    level=logging.INFO) 
                

if __name__ == "__main__":
    logging.info("Logging has been started")
    logging.info("Logging has been ended")

