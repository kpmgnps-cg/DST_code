import logging as logger
import sys

root = logger.getLogger()
root.setLevel(logger.DEBUG)
logger.getLogger("py4j").setLevel(logger.ERROR)

while root.handlers:
  root.handlers.pop() 

handler = logger.StreamHandler(sys.stdout)
handler.setLevel(logger.DEBUG)
formatter = logger.Formatter('%(asctime)s -%(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
root.addHandler(handler)
logger.propagate = False