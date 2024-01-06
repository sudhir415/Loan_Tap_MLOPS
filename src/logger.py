import logging
import os
from datetime import datetime 

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" 
LOG_PATH=os.path.join(os.getcwd(),"logs", LOG_FILE)
os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH=os.path.join(LOG_PATH, LOG_FILE)

logging.basicConfig(filename=LOG_FILE_PATH, format='%(asctime)s %(levelname)s %(name)s %(message)s', level=logging.INFO)
                

# logger = logging.getLogger(__name__) 

# # Example usage of the logger
# logger.debug('This is a debug message')
# logger.info('This is an info message')
# logger.warning('This is a warning message')
# logger.error('This is an error message')
# logger.critical('This is a critical message') 

