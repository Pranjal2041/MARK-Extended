# An Advanced Logger class which writes data in a well formatted manner 
# to files based on different priorities.

from enum import Enum
import os
import datetime
import time

class LogPriority(Enum):
    """
    Enum class for different log priorities.
    """
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    STATS = 3

class AdvancedLogger:

    def __init__(self, base_dir):
        self.base_dir =  base_dir
        self.files = []
        self.file_names = []
        for p in LogPriority:
            self.file_names.append(os.path.join(self.base_dir, f'Log_{p.name}' + '.log'))
            self.files.append(open(self.file_names[-1], 'w'))
        self.last_log_time = -1

    def flush(self):
        for f in self.files:
                f.close()
        for i in range(len(self.files)):
            self.files[i] = open(self.file_names[i], 'a')

    def log(self, *args, priority = LogPriority.LOW):
        to_log = ' '.join(map(str, args))
        if priority.value <= LogPriority.MEDIUM.value:
            # Add current time to to_log
            now = datetime.datetime.now()
            to_log = f'[{now.strftime("%H:%M:%S")}]: {to_log}' 
        print(to_log)
        for p in range(priority.value+1):
            self.files[p].write(to_log + '\n')
        
        # If time - last_log_time is greater than 10s or Priority is HIGH or above close the file and re-open in append mode
        if time.time() - self.last_log_time > 10 or priority.value >= LogPriority.HIGH.value: 
            self.flush()
            self.last_log_time = time.time()